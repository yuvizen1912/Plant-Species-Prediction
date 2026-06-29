import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- GLOBAL VARIABLES FOR THE MODEL ---
clf = None
X_labels = []

def load_and_train():
    global clf, X_labels
    
    # Hide the main window while choosing the file
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select your Iris.csv file", 
                                           filetypes=[("CSV files", "*.csv")])
    
    if not file_path:
        messagebox.showerror("Error", "No file selected. Exiting application.")
        root.destroy()
        return

    try:
        # Load Data
        df = pd.read_csv(file_path)
        
        # Clean columns (handle case variations)
        df.columns = [col.strip() for col in df.columns]
        
        # Preprocessing
        X = df.drop(columns=['Id', 'Species'], errors='ignore')
        X_labels = X.columns.tolist()
        
        # Map target
        y_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        Y = df['Species'].map(y_mapping)
        
        # Train-Test Split & Train
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Metrics
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=list(y_mapping.keys()))
        
        # Bring back and build the GUI
        root.deiconify()
        setup_gui(df.head().to_string(), acc, report, X_labels)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        root.destroy()

def make_prediction(entries):
    try:
        # 1. Extract values from UI textboxes
        input_data = [float(entries[label].get()) for label in X_labels]
        
        # 2. Convert the input list into a DataFrame with matching feature names
        # This resolves the "X does not have valid feature names" warning
        input_df = pd.DataFrame([input_data], columns=X_labels)
        
        # 3. Predict using the DataFrame instead of a raw list
        pred_encoded = clf.predict(input_df)[0]
        
        species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        result = species_map[pred_encoded]
        
        # Display output dynamic popup
        messagebox.showinfo("Prediction Result", f"The model predicts this flower is: {result}")
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter valid numbers for all measurements.")

def setup_gui(df_head_str, accuracy, report_str, features):
    root.title("Iris Model Analytics & Predictor Hub")
    root.geometry("650x700")
    
    # Notebook/Tabs layout
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill='both')
    
    # --- TAB 1: MODEL PERFORMANCE ---
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="📊 Model Metrics")
    
    tk.Label(tab1, text=f"Global Accuracy: {accuracy * 100:.2f}%", font=("Arial", 14, "bold"), fg="green").pack(pady=10)
    
    tk.Label(tab1, text="Classification Report:", font=("Arial", 11, "bold")).pack(anchor='w', padx=10)
    report_text = tk.Text(tab1, height=12, width=70, font=("Courier", 10))
    report_text.insert(tk.END, report_str)
    report_text.config(state=tk.DISABLED) # Read-only
    report_text.pack(pady=5, padx=10)
    
    tk.Label(tab1, text="Data Sample (df.head):", font=("Arial", 11, "bold")).pack(anchor='w', padx=10)
    df_text = tk.Text(tab1, height=8, width=70, font=("Courier", 10))
    df_text.insert(tk.END, df_head_str)
    df_text.config(state=tk.DISABLED)
    df_text.pack(pady=5, padx=10)

    # --- TAB 2: INTERACTIVE PREDICTOR ---
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="🔮 Live Predictor")
    
    tk.Label(tab2, text="Enter Flower Measurements to Test Model:", font=("Arial", 12, "bold")).pack(pady=15)
    
    # Dynamically generate input fields based on dataset columns
    entries = {}
    frame_inputs = ttk.Frame(tab2)
    frame_inputs.pack(pady=10)
    
    for idx, feature in enumerate(features):
        lbl = tk.Label(frame_inputs, text=f"{feature}:", font=("Arial", 10))
        lbl.grid(row=idx, column=0, padx=10, pady=8, sticky="e")
        entry = tk.Entry(frame_inputs, width=15)
        entry.grid(row=idx, column=1, padx=10, pady=8)
        # Default placeholder value for easy testing
        entry.insert(0, "5.1" if "length" in feature.lower() else "3.5") 
        entries[feature] = entry
        
    # Interactive Button
    predict_btn = tk.Button(tab2, text="⚡ Run Inference", bg="#4CAF50", fg="white", 
                            font=("Arial", 11, "bold"), command=lambda: make_prediction(entries), padx=10, pady=5)
    predict_btn.pack(pady=20)

# --- INITIALIZATION ---
root = tk.Tk()
root.attributes('-topmost', True)

# Start by triggering file loader
root.after(100, load_and_train)
root.mainloop()