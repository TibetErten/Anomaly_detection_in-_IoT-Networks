import os


def save_output_to_file(content, file_name="output.txt", directory=None):
    """
    Saves the given content to a text file.

    Args:
        content (str): The content to save to the file.
        file_name (str): The name of the file (default is 'output.txt').
        directory (str): The directory where the file should be saved.

    Returns:
        str: Full path to the saved file.
    """
    # If a directory is specified, ensure it exists
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)
    else:
        file_path = file_name

    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(content)

    print(f"Output saved to {file_path}")
    return file_path


def save_plot_to_file(fig, file_name="plot.png", directory=None):
    """
    Saves a Matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure to save.
        file_name (str): The name of the file (default is 'plot.png').
        directory (str): The directory where the file should be saved.

    Returns:
        str: Full path to the saved file.
    """
    # If a directory is specified, ensure it exists
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)
    else:
        file_path = file_name

    # Save the figure to the file
    fig.savefig(file_path)

    print(f"Plot saved to {file_path}")
    return file_path


