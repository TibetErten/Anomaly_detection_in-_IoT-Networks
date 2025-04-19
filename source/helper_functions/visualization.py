import matplotlib.pyplot as plt
from source.helper_functions.save_outputs import save_plot_to_file, save_output_to_file
from io import StringIO



def plot_benign_vs_others(data, column_name, save_dir):
    """
    Generates a pie chart for the distribution of 'BENIGN' vs. 'Attack' labels.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to analyze.
        save_dir (str): The directory where the pie chart will be saved.

    Returns:
        None
    """
    if column_name in data.columns:
        # Create a new column to classify BENIGN vs. Attack
        data['Category'] = data[column_name].apply(
            lambda x: 'BENIGN' if isinstance(x, str) and 'BENIGN' in x.upper() else 'Attack'
        )

        # Pie chart for BENIGN vs. Attack
        category_counts = data['Category'].value_counts()
        plt.figure(figsize=(8, 8))
        category_counts.plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            cmap='Set3',
            colors=plt.cm.Set3.colors[:len(category_counts)]
        )
        plt.title("BENIGN vs. Attack Distribution")
        plt.ylabel("")  # Remove y-axis label for better visualization
        save_plot_to_file(plt.gcf(), "benign_vs_attack_pie_chart.png", save_dir)
        plt.show()
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")

    
def plot_attack_type_distribution(data, column_name, save_dir):
    """
    Generates a bar chart for the distribution of attack types with counts displayed on the bars.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to analyze.
        save_dir (str): The directory where the bar chart will be saved.

    Returns:
        None
    """
    if column_name in data.columns:
        # Filter only attack rows
        attack_data = data[data[column_name].str.upper() != 'BENIGN']

        # Count attack types
        attack_type_counts = attack_data[column_name].value_counts()
        print(f"Attack type counts:\n{attack_type_counts}")

        # Generate the bar chart
        plt.figure(figsize=(12, 6))
        ax = attack_type_counts.plot.bar(color='skyblue', edgecolor='black')
        plt.title("Attack Types Distribution")
        plt.xlabel("Attack Types")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

        # Add numbers on top of the bars
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),  # The height of the bar (count)
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position at the center-top of the bar
                ha='center', va='bottom', fontsize=10, color='black'
            )

        plt.tight_layout()

        # Save the bar chart
        save_plot_to_file(plt.gcf(), "attack_types_bar_chart.png", save_dir)

        # Show the bar chart
        plt.show()
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


def analyze_data_quality(data, save_dir):
    """
    Analyzes the data quality by finding duplicate rows, missing values, and generating pie charts.

    Args:
        data (pd.DataFrame): The DataFrame to analyze.
        save_dir (str): The directory where outputs (reports and charts) will be saved.

    Returns:
        None
    """

    # Save data info to a file
    buffer = StringIO()
    data.info(buf=buffer)
    data_info = buffer.getvalue()
    save_output_to_file(data_info, 'data_info.txt', save_dir)

    # Find the number of duplicate rows
    num_duplicates = data.duplicated().sum()
    total_rows = len(data)
    duplicate_percentage = (num_duplicates / total_rows) * 100
    print(f"Number of duplicate rows: {num_duplicates} ({duplicate_percentage:.2f}%)")

    # Pie chart for duplicate rows
    plt.figure(figsize=(6, 6))
    plt.pie(
        [num_duplicates, total_rows - num_duplicates],
        labels=['Duplicates', 'Unique Rows'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['red', 'green']
    )
    plt.title("Duplicate Rows Percentage")
    save_plot_to_file(plt.gcf(), "duplicate_rows_pie_chart.png", save_dir)
    plt.show()

    # Find the number of missing values per column
    missing_values = data.isnull().sum()
    total_cells = data.size
    missing_cells = missing_values.sum()
    missing_percentage = (missing_cells / total_cells) * 100
    print("Number of missing values per column:")
    print(missing_values)
    print(f"Total missing values: {missing_cells} ({missing_percentage:.2f}%)")

    # Pie chart for missing values
    plt.figure(figsize=(6, 6))
    plt.pie(
        [missing_cells, total_cells - missing_cells],
        labels=['Missing Values', 'Non-Missing Values'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['orange', 'blue']
    )
    plt.title("Missing Values Percentage")
    save_plot_to_file(plt.gcf(), "missing_values_pie_chart.png", save_dir)
    plt.show()

    # Save the results to a file
    duplicates_info = f"Number of duplicate rows: {num_duplicates} ({duplicate_percentage:.2f}%)\n"
    missing_values_info = (
        "Number of missing values per column:\n" + missing_values.to_string() +
        f"\nTotal missing values: {missing_cells} ({missing_percentage:.2f}%)"
    )
    save_output_to_file(duplicates_info + missing_values_info, 'data_quality_report.txt', save_dir)