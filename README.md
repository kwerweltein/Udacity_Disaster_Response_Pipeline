# udacity-disaster-response-pipeline

T.Bender for udacity


## About The Project


## Motivation
The primary motivation behind this project was to apply data analysis techniques to a real-world dataset to extract meaningful insights. Specifically, we aimed to:
*   Understand the distribution of Airbnb prices in Athens.
*   Identify the most expensive and most affordable neighborhoods.
*   Visualize geographical distribution of listings and prices.


## Libraries Used

This project leverages the following Python libraries for data manipulation, analysis, and visualization:

**Data Cleaning & Calculation:**
*   `pandas` (for data manipulation and analysis)
*   `numpy` (for numerical operations)
*   `math` (for mathematical functions)

**Plotting & Visualization:**
*   `matplotlib.pyplot` (for static and basic plots)
*   `plotly.express` (for interactive, high-level plots)
*   `plotly.graph_objects` (for fine-grained control over plots)
*   `plotly.subplots` (for creating subplots with Plotly)
*   `seaborn` (for statistical data visualization, built on Matplotlib)

**Map Creation:**
*   `folium` (for interactive geospatial visualizations)
*   `branca.colormap` (for creating colormaps for Folium maps)
*   `os` (for operating system functionalities, e.g., path manipulation)

## Repository Structure
```
├── data/
    ├── messages.csv
    ├── categories.csv
    ├── ETL Pipeline Preparation.ipynb : preparation notebook
    └── process_data.py
├── app/ : 
├── maps/ : Contains the interactive HTML maps
├── Athens_Airbnb_Notebook.ipynb
└── README.md
```

## Results Summary

A summary of the project's findings and key visualizations are deployed and accessible online:

*   **Project Summary:** [https://kwerweltein.github.io/](https://kwerweltein.github.io/)


## Acknowledgments
*   **Inside Airbnb** ([https://insideairbnb.com/get-the-data/](https://insideairbnb.com/get-the-data/)) for providing the comprehensive and publicly available Airbnb dataset for Athens.