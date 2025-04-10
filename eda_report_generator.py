import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.bbox"] = "tight"

class EDAReportGenerator:

    @staticmethod
    def _add_page(pdf, fig):
        pdf.savefig(fig)
        plt.close(fig)

    @staticmethod
    def _dataset_overview(df: pd.DataFrame, pdf: PdfPages):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Dataset Overview', fontsize=16, loc='left')

        lines = [
            f"Row Count: {len(df)}",
            f"Column Count: {len(df.columns)}",
            "", "Columns:"
        ] + [f"- {col}" for col in df.columns]

        lines += ["", "Data Types:"]
        lines += [f"- {col}: {df[col].dtype}" for col in df.columns]

        for i, line in enumerate(lines):
            ax.text(0, 1 - (i + 1) * 0.035, line, fontsize=11, transform=ax.transAxes)

        EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def _summary_statistics(df: pd.DataFrame, pdf: PdfPages):
        summary = df.describe().round(2).T
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Summary Statistics', fontsize=16, loc='left')

        ax.text(0, 1 - 0.05, " | ".join(summary.columns), fontsize=10, transform=ax.transAxes)
        for i, (col, row) in enumerate(summary.iterrows()):
            line = f"{col}: " + " | ".join(str(val) for val in row.values)
            ax.text(0, 1 - (i + 2) * 0.035, line, fontsize=9, transform=ax.transAxes)

        EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def _missing_values(df: pd.DataFrame, pdf: PdfPages):
        missing = df.isnull().sum()
        perc = (missing / len(df)) * 100
        missing_df = pd.DataFrame({"Missing": missing, "Percentage": perc}).sort_values(by="Percentage", ascending=False)

        # Text
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Missing Values Summary', fontsize=16, loc='left')
        lines = [
            f"{col}: {row['Missing']} ({row['Percentage']:.2f}%)"
            for col, row in missing_df.iterrows() if row["Missing"] > 0
        ] or ["No missing values found."]

        for i, line in enumerate(lines):
            ax.text(0, 1 - (i + 1) * 0.045, line, fontsize=11, transform=ax.transAxes)

        EDAReportGenerator._add_page(pdf, fig)

        # Heatmap
        fig = plt.figure()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def _univariate_analysis(df: pd.DataFrame, pdf: PdfPages):
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

        for col in num_cols:
            fig = plt.figure()
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            EDAReportGenerator._add_page(pdf, fig)

        for col in cat_cols:
            fig = plt.figure()
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            plt.xticks(rotation=45)
            plt.title(f'Categories of {col}')
            EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def _correlation_analysis(df: pd.DataFrame, pdf: PdfPages):
        corr = df.corr(numeric_only=True)

        # Heatmap
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        EDAReportGenerator._add_page(pdf, fig)

        # High correlation pairs
        threshold = 0.8
        high_corr = corr[(corr.abs() > threshold) & (corr.abs() < 1)].stack().reset_index()
        high_corr.columns = ["Feature 1", "Feature 2", "Correlation"]
        pairs = high_corr.to_dict(orient="records")

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title("Highly Correlated Pairs", fontsize=16, loc='left')

        if not pairs:
            ax.text(0, 0.95, "No high correlations found.", fontsize=12, transform=ax.transAxes)
        else:
            for i, row in enumerate(pairs):
                ax.text(0, 1 - (i + 1) * 0.045, f"{row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.2f}", fontsize=11, transform=ax.transAxes)

        EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def _outlier_analysis(df: pd.DataFrame, pdf: PdfPages):
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            fig = plt.figure()
            sns.boxplot(data=df, x=col)
            plt.title(f'Boxplot of {col}')
            EDAReportGenerator._add_page(pdf, fig)

    @staticmethod
    def generate(df, output_path: str):
        with PdfPages(output_path) as pdf:
            EDAReportGenerator._dataset_overview(df, pdf)
            EDAReportGenerator._summary_statistics(df, pdf)
            EDAReportGenerator._missing_values(df, pdf)
            EDAReportGenerator._univariate_analysis(df, pdf)
            EDAReportGenerator._correlation_analysis(df, pdf)
            EDAReportGenerator._outlier_analysis(df, pdf)

