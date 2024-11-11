from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uuid
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import threading
import time


app = FastAPI()

# Directory to save plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory="plots"), name="plots")


# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from this domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for datasets (to handle multiple users simultaneously)
datasets: Dict[str, pd.DataFrame] = {}

def clear_old_plots():
    while True:
        try:
            current_time = time.time()
            for filename in os.listdir(PLOT_DIR):
                file_path = os.path.join(PLOT_DIR, filename)
                if os.path.isfile(file_path):
                    # Remove files older than 5 minute (300 seconds)
                    if current_time - os.path.getmtime(file_path) > 300:
                        os.remove(file_path)
        except Exception as e:
            print(f"Error while clearing old plots: {e}")
        time.sleep(300)  # Run this cleanup every five minutes

# Start the thread to clear old plots
threading.Thread(target=clear_old_plots, daemon=True).start()

def standard_response(success: bool, message: str, data: dict = None):
    return {
        "success": success,
        "message": message,
        "data": data
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Generate a unique ID for the dataset
        dataset_id = str(uuid.uuid4())
        # Read dataset from CSV or Excel file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Store the dataset in-memory with the unique ID
        datasets[dataset_id] = df
        
        # Example analysis: show dataset information
        dataset_info = {
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "row_count": len(df),
            "column_info": {
                col: str(df[col].dtype) for col in df.columns
            }, 
            "summary": df.describe().to_dict()
        
        }

        # print(f'Upload dataset response: {dataset_info}')
        return standard_response(True, "Dataset uploaded successfully.", dataset_info)
    except Exception as e:
        print(f'Exception handling upload api: {e}')
        return standard_response(False, str(e))


@app.get("/analyze/{dataset_id}/missing_values")
async def analyze_missing_values(dataset_id: str):
    try:
        # Check if dataset exists
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Retrieve the dataset
        df = datasets[dataset_id]
        
        # Analyze missing values
        missing_values = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / df.shape[0]) * 100
        missing_df = pd.DataFrame({'missing_values': missing_values, 'percentage': missing_percentage})
        response = missing_df[missing_df['missing_values'] > 0].sort_values(by='percentage', ascending=False).to_dict()

        # Visualize missing values using a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        
        # Save plot to a file
        plot_file = os.path.join(PLOT_DIR, f"missing_values_{uuid.uuid4()}.png")
        plt.savefig(plot_file)
        plt.close()

        # Return the plot as a URL that can be accessed
        plot_url = f"plots/{os.path.basename(plot_file)}"
        response['heatmap'] = plot_url

        return standard_response(True, "Missing values analysis complete.", response)
    except Exception as e:
        return standard_response(False, str(e))
    
@app.get("/analyze/{dataset_id}/univariate_analysis")
async def univariate_analysis(dataset_id: str):  
    try:
        # Check if dataset exists
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Retrieve the dataset
        df = datasets[dataset_id]
        
        # Select numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        plot_files = []

        # Plot distribution for each numerical column in the current page
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

            # Save plot to a file
            plot_file = os.path.join(PLOT_DIR, f"univariate_{col}_{uuid.uuid4()}.png")
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(f"plots/{os.path.basename(plot_file)}")

        # Create HTML response with pagination
        return standard_response(True, "Univariate analysis complete.", {"plots": plot_files})
    except Exception as e:
        return standard_response(False, str(e))
    
@app.get("/analyze/{dataset_id}/categorical_univariate_analysis")
async def univariate_analysis(dataset_id: str):  
    try:
        # Check if dataset exists
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Retrieve the dataset
        df = datasets[dataset_id]
        
        # Select numerical columns
        num_cols = df.select_dtypes(include=['object']).columns
        plot_files = []

        # Plot distribution for each numerical column in the current page
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            # plt.title(f'Count of Categories in {col}')
            plt.xticks(rotation=45)
            plt.title(f'Categories of {col}')
            plt.ylabel('Frequency')

            # Save plot to a file
            plot_file = os.path.join(PLOT_DIR, f"categorical_univariate_{col}_{uuid.uuid4()}.png")
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(f"plots/{os.path.basename(plot_file)}")

        # Create HTML response with pagination
        return standard_response(True, "Categorical univariate analysis complete.", {"plots": plot_files})
    except Exception as e:
        return standard_response(False, str(e))

@app.get("/analyze/{dataset_id}/correlation")
async def analyze_correlation(dataset_id: str):
    try:
        # Check if dataset exists
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Retrieve the dataset
        df = datasets[dataset_id]
        
        # Calculate correlation matrix
        correlation_matrix = df.corr(numeric_only=True)
        
        # Plot heatmap for correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        
        # Save plot to a file
        plot_file = os.path.join(PLOT_DIR, f"missing_values_{uuid.uuid4()}.png")
        plt.savefig(plot_file)
        plt.close()

        # Return the plot as a URL that can be accessed
        plot_url = f"plots/{os.path.basename(plot_file)}"
        
        # Display highly correlated pairs
        threshold = 0.8
        high_corr = correlation_matrix[(correlation_matrix.abs() > threshold) & (correlation_matrix.abs() < 1.0)].stack().reset_index()
        high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        high_corr_pairs = high_corr.to_dict(orient='records')
        
        return standard_response(True, "Correlation analysis complete.", {
            "heatmap": plot_url,
            "highly_correlated_pairs": high_corr_pairs
        })
    except Exception as e:
        return standard_response(False, str(e))

@app.get("/analyze/{dataset_id}/outlier_plots")
async def analyze_unique_values(dataset_id: str):
    try:
        # Check if dataset exists
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Retrieve the dataset
        df = datasets[dataset_id]

         # Select numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        plot_files = []

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=col)
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.ylabel('Values')
            
            # Save plot to a file
            plot_file = os.path.join(PLOT_DIR, f"outlier_{col}_{uuid.uuid4()}.png")
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(f"plots/{os.path.basename(plot_file)}")

        return standard_response(True, "Boxplots to visualize outliers in numerical columns", {"plots": plot_files})
    except Exception as e:
        return standard_response(False, str(e))

# @app.get("/analyze/{dataset_id}/report")
# async def analyze_value_counts(dataset_id: str, column: str):
#     try:
#         # Check if dataset exists
#         if dataset_id not in datasets:
#             raise HTTPException(status_code=404, detail="Dataset not found.")
        
#         # Retrieve the dataset
#         df = datasets[dataset_id]
        
#         # TODO: Use LLM to generate a text report of the dataset
#         return standard_response(True, "Value counts analysis complete.", {"report": ''})
#     except Exception as e:
#         return standard_response(False, str(e))