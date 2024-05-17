# Author: Elise Zhang
# Date: 15 mai 2024

# Environment 'DAGCompare' created with conda
# conda create -n DAGCompare python=3.9
# conda activate DAGCompare
# pip install numpy


import tkinter as tk
import numpy as np

def calculate_accuracy(adj_matrix_gt, adj_matrix_pred):
    # correctly predicted positive (1) entry + correctly predicted negative (0) entry / total number of entries
    num_correct = np.sum(adj_matrix_gt == adj_matrix_pred)
    accuracy = num_correct / adj_matrix_gt.size
    return accuracy

def calculate_TP_FP_TN_FN(adj_matrix_gt, adj_matrix_pred):
    # True Positive: 1 in GT and 1 in Pred
    # False Positive: 0 in GT and 1 in Pred
    # True Negative: 0 in GT and 0 in Pred
    # False Negative: 1 in GT and 0 in Pred
    TP = np.sum(np.logical_and(adj_matrix_gt == 1, adj_matrix_pred == 1))
    FP = np.sum(np.logical_and(adj_matrix_gt == 0, adj_matrix_pred == 1))
    TN = np.sum(np.logical_and(adj_matrix_gt == 0, adj_matrix_pred == 0))
    FN = np.sum(np.logical_and(adj_matrix_gt == 1, adj_matrix_pred == 0))
    return TP, FP, TN, FN

def calculate_precision(adj_matrix_gt, adj_matrix_pred):
    num_true_positive = np.sum(np.logical_and(adj_matrix_gt == 1, adj_matrix_pred == 1))
    num_predicted_positive = np.sum(adj_matrix_pred)
    precision = num_true_positive / max(num_predicted_positive, 1)
    return precision

def calculate_recall(adj_matrix_gt, adj_matrix_pred):
    num_true_positive = np.sum(np.logical_and(adj_matrix_gt == 1, adj_matrix_pred == 1))
    num_actual_positive = np.sum(adj_matrix_gt)
    recall = num_true_positive / max(num_actual_positive, 1)
    return recall

def calculate_f1_score(adj_matrix_gt, adj_matrix_pred):
    precision = calculate_precision(adj_matrix_gt, adj_matrix_pred)
    recall = calculate_recall(adj_matrix_gt, adj_matrix_pred)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
    return f1_score

def calculate_structural_hamming_distance(adj_matrix_gt, adj_matrix_pred):
    num_mismatch_edges = np.sum(adj_matrix_gt != adj_matrix_pred)
    shd = num_mismatch_edges 
    return shd

class DAGComparisonToolbox:
    def __init__(self, root):
        self.root = root
        self.root.title("DAG Comparison Toolbox")
        
        # Variables
        self.num_variables = tk.IntVar(value=4)  # Default to 4 variables
        self.graph_left = None
        self.graph_right = None
        self.adj_matrix_left = None
        self.adj_matrix_right = None
        self.nodes_left = []
        self.nodes_right = []
        self.edges_left = []
        self.edges_right = []
        
        # Create GUI elements
        self.create_interface()
        
    def create_interface(self):
        # Label and entry for number of variables
        lbl_num_variables = tk.Label(self.root, text="Number of Variables:")
        lbl_num_variables.pack()
        entry_num_variables = tk.Entry(self.root, textvariable=self.num_variables)
        entry_num_variables.pack()
        
        # Button to generate DAGs
        btn_generate_dags = tk.Button(self.root, text="Generate DAGs", command=self.generate_dags)
        btn_generate_dags.pack()

        # Button to reset left graph
        btn_reset_left = tk.Button(self.root, text="Reset Left Canvas", command=self.reset_left_gt_canvas)
        btn_reset_left.pack()

        # Button to reset right graph
        btn_reset_right = tk.Button(self.root, text="Reset Right Canvas", command=self.reset_right_pred_canvas)
        btn_reset_right.pack()
        
        # Frame for left and right graphs
        self.frame_left = tk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.frame_right = tk.Frame(self.root)
        self.frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Canvas for left and right graphs
        self.canvas_left = tk.Canvas(self.frame_left, bg="white")
        self.canvas_left.pack(expand=True, fill=tk.BOTH)
        self.canvas_left.bind("<Button-1>", self.select_node_left)
        self.canvas_left.create_text(10, 10, anchor="nw", text="Ground Truth", fill="black")
        
        self.canvas_right = tk.Canvas(self.frame_right, bg="white")
        self.canvas_right.pack(expand=True, fill=tk.BOTH)
        self.canvas_right.bind("<Button-1>", self.select_node_right)
        self.canvas_right.create_text(10, 10, anchor="nw", text="Predicted Graph", fill="black")
        
        # Button to display adjacency matrix
        btn_display_matrix = tk.Button(self.root, text="Display Adjacency Matrix", command=self.display_adjacency_matrix)
        btn_display_matrix.pack()

        # Button to calculate metric scores
        btn_calculate_metrics = tk.Button(self.root, text="Calculate Metrics", command=self.calculate_metrics)
        btn_calculate_metrics.pack()

        # Text elements for displaying matrices and scores
        self.text_adj_matrix_left = self.canvas_left.create_text(200, 200, anchor="nw", fill="black")
        self.text_adj_matrix_right = self.canvas_right.create_text(200, 200, anchor="nw", fill="black")

        # Text element for displaying metric scores
        self.canvas_scores = tk.Canvas(self.root, bg="white", width=300, height=300)
        self.canvas_scores.pack()
        self.text_metric_scores = self.canvas_scores.create_text(50, 50, text="", font=("Arial", 14), fill="black")



    def display_adjacency_matrix(self):
        # if self.adj_matrix_left is not None and self.adj_matrix_right is not None:
        #     # Update text elements with matrices
        #     self.canvas_left.itemconfig(self.text_adj_matrix_left, text="Adjacency Matrix - Ground Truth:\n" + str(self.adj_matrix_left))
        #     self.canvas_right.itemconfig(self.text_adj_matrix_right, text="Adjacency Matrix - Predicted:\n" + str(self.adj_matrix_right))
        if self.adj_matrix_left is not None and self.adj_matrix_right is not None:
            # Convert adjacency matrices to strings for display
            matrix_left_str = "\n".join([" ".join(map(str, row)) for row in self.adj_matrix_left])
            matrix_right_str = "\n".join([" ".join(map(str, row)) for row in self.adj_matrix_right])

            # Clear previous text in the text elements
            self.canvas_left.delete(self.text_adj_matrix_left)
            self.canvas_right.delete(self.text_adj_matrix_right)

            # Insert new text into the text elements
            self.text_adj_matrix_left = self.canvas_left.create_text(200, 20, anchor="nw", text="Adjacency Matrix \n Ground Truth:\n" + matrix_left_str, fill="black", font=("Arial", 16) )
            self.text_adj_matrix_right = self.canvas_right.create_text(200, 20, anchor="nw", text="Adjacency Matrix \n Predicted:\n" + matrix_right_str, fill="black", font=("Arial", 16) )

            # also display in terminal
            print("Adjacency Matrix - Ground Truth:")
            print(str(self.adj_matrix_left)+"\n")
            print("Adjacency Matrix - Predicted:")
            print(str(self.adj_matrix_right)+"\n")
        else:
            print("Adjacency matrices are not yet generated.")


        
    def generate_dags(self):
        num_vars = self.num_variables.get()
        
        # Clear previous graphs, nodes, edges, and adjacency matrices
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        self.canvas_left.create_text(10, 10, anchor="nw", text="Ground Truth", fill="black")
        self.canvas_right.create_text(10, 10, anchor="nw", text="Predicted Graph", fill="black")
        self.nodes_left = []
        self.nodes_right = []
        self.temp_current_edge_left = [] # to store node pairs, once we have two nodes, we can draw an edge, and clear the temp
        self.temp_current_edge_right = [] # to store node pairs, once we have two nodes, we can draw an edge, and clear the temp
        self.edges_left = [] # to store node pairs
        self.edges_right = [] # to store node pairs
        self.adj_matrix_left = np.zeros((num_vars, num_vars))
        self.adj_matrix_right = np.zeros((num_vars, num_vars))

        # also clear the metrics display
        self.canvas_scores.delete("all")

        # Adjust canvas size based on number of variables
        canvas_width = 50 + 160 * 2  # Circle diameter + padding
        canvas_height = 50 + (num_vars - 1) * 50 + 160  # Circle diameter + (num_vars - 1) * spacing + padding
        self.canvas_left.config(width=canvas_width, height=canvas_height)
        self.canvas_right.config(width=canvas_width, height=canvas_height)
        
        # Draw circles for variables on left and right with names inside
        left_x = 50
        right_x = 50
        y_spacing = 50
        for i in range(num_vars):
            y = 50 + i * y_spacing
            name = f"Y{i}"
            self.canvas_left.create_oval(left_x, y, left_x + 30, y + 30, outline="black", width=2, tags=f"node_left_{i}")
            self.canvas_left.create_text(left_x + 15, y + 15, text=name, fill="black")
            self.nodes_left.append((left_x + 15, y + 15))
            
            self.canvas_right.create_oval(right_x, y, right_x + 30, y + 30, outline="black", width=2, tags=f"node_right_{i}")
            self.canvas_right.create_text(right_x + 15, y + 15, text=name, fill="black")
            self.nodes_right.append((right_x + 15, y + 15))
        
    def reset_left_gt_canvas(self):
        num_vars = self.num_variables.get()
        self.canvas_left.delete("all")
        self.canvas_left.create_text(10, 10, anchor="nw", text="Ground Truth", fill="black")
        self.nodes_left = []
        self.temp_current_edge_left = []
        self.edges_left = []
        self.adj_matrix_left = np.zeros((num_vars, num_vars))
        # also clear the metrics display
        self.canvas_scores.delete("all")

        # Adjust canvas size based on number of variables
        canvas_width = 50 + 160 * 2  # Circle diameter + padding
        canvas_height = 50 + (num_vars - 1) * 50 + 160  # Circle diameter + (num_vars - 1) * spacing + padding
        self.canvas_left.config(width=canvas_width, height=canvas_height)

        # Draw circles for variables on left and right with names inside
        left_x = 50
        y_spacing = 50
        for i in range(num_vars):
            y = 50 + i * y_spacing
            name = f"Y{i}"
            self.canvas_left.create_oval(left_x, y, left_x + 30, y + 30, outline="black", width=2, tags=f"node_left_{i}")
            self.canvas_left.create_text(left_x + 15, y + 15, text=name, fill="black")
            self.nodes_left.append((left_x + 15, y + 15))


    def reset_right_pred_canvas(self):
        num_vars = self.num_variables.get()
        self.canvas_right.delete("all")
        self.canvas_right.create_text(10, 10, anchor="nw", text="Predicted Graph", fill="black")

        self.nodes_right = []
        self.temp_current_edge_right = []
        self.edges_right = []
        self.adj_matrix_right = np.zeros((num_vars, num_vars))
        # also clear the metrics display
        self.canvas_scores.delete("all")

        # Adjust canvas size based on number of variables
        canvas_width = 50 + 160 * 2  # Circle diameter + padding
        canvas_height = 50 + (num_vars - 1) * 50 + 160  # Circle diameter + (num_vars - 1) * spacing + padding
        self.canvas_right.config(width=canvas_width, height=canvas_height)
        
        # Draw circles for variables on left and right with names inside
        right_x = 50
        y_spacing = 50
        for i in range(num_vars):
            y = 50 + i * y_spacing
            name = f"Y{i}"
            self.canvas_right.create_oval(right_x, y, right_x + 30, y + 30, outline="black", width=2, tags=f"node_right_{i}")
            self.canvas_right.create_text(right_x + 15, y + 15, text=name, fill="black")
            self.nodes_right.append((right_x + 15, y + 15))


    def select_node_left(self, event):
        x, y = event.x, event.y
        node = self.find_closest_node(x, y, self.nodes_left)
        if node:
            if len(self.temp_current_edge_left) < 2:
                self.temp_current_edge_left.append(node)
                if len(self.temp_current_edge_left) == 2:
                    self.edges_left.append(self.temp_current_edge_left)
                    self.draw_edge(self.canvas_left, self.temp_current_edge_left)
                    self.update_adj_matrix(self.adj_matrix_left, self.temp_current_edge_left)
                    self.temp_current_edge_left = []
        
    def select_node_right(self, event):
        x, y = event.x, event.y
        node = self.find_closest_node(x, y, self.nodes_right)
        if node:
            if len(self.temp_current_edge_right) < 2:
                self.temp_current_edge_right.append(node)
                if len(self.temp_current_edge_right) == 2:
                    self.edges_right.append(self.temp_current_edge_right)
                    self.draw_edge(self.canvas_right, self.temp_current_edge_right)
                    self.update_adj_matrix(self.adj_matrix_right, self.temp_current_edge_right)
                    self.temp_current_edge_right = []
        

    def find_closest_node(self, x, y, nodes):
        min_dist = float("inf")
        closest_node = None
        for node_x, node_y in nodes:
            dist = (x - node_x)**2 + (y - node_y)**2
            if dist < min_dist:
                min_dist = dist
                closest_node = (node_x, node_y)
        if min_dist <= 1000:  # Square of maximum distance 
            return closest_node
        else:
            return None

    def draw_edge(self, canvas, current_edge):
        # if the two nodes connected are adjacent in indices, draw a straight arrow from one to the other
        # otherwise, draw a curved arrow
        index1 = self.nodes_left.index(current_edge[0]) if current_edge[0] in self.nodes_left else self.nodes_right.index(current_edge[0])
        index2 = self.nodes_left.index(current_edge[1]) if current_edge[1] in self.nodes_left else self.nodes_right.index(current_edge[1])
        if abs(index1 - index2) == 1:
            x1, y1 = current_edge[0]
            x2, y2 = current_edge[1]
            canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill="red", width=2, tags="edge")
        else:
            x1, y1 = current_edge[0]
            x2, y2 = current_edge[1]
            control_x = (x1 + x2) // 2 + 30*abs(index1 - index2)
            control_y = (y1 + y2) // 2 
            canvas.create_line(x1, y1, control_x, control_y, x2, y2, arrow=tk.LAST, smooth=True, fill="red", width=2, tags="edge")
        
    def update_adj_matrix(self, adj_matrix, nodes):
        if len(nodes) == 2:
            index1 = self.nodes_left.index(nodes[0]) if nodes[0] in self.nodes_left else self.nodes_right.index(nodes[0])
            index2 = self.nodes_left.index(nodes[1]) if nodes[1] in self.nodes_left else self.nodes_right.index(nodes[1])
            adj_matrix[index1][index2] = 1
        
    def calculate_metrics(self):
        # Calculate the five metrics
        self.accuracy = calculate_accuracy(self.adj_matrix_left, self.adj_matrix_right)
        self.TP, self.FP, self.TN, self.FN = calculate_TP_FP_TN_FN(self.adj_matrix_left, self.adj_matrix_right)
        self.precision = calculate_precision(self.adj_matrix_left, self.adj_matrix_right)
        self.recall = calculate_recall(self.adj_matrix_left, self.adj_matrix_right)
        self.f1_score = calculate_f1_score(self.adj_matrix_left, self.adj_matrix_right)
        self.shd = calculate_structural_hamming_distance(self.adj_matrix_left, self.adj_matrix_right)

        # display in canvas
        scores_text_1 = f"Accuracy: {self.accuracy:.4f}\nPrecision: {self.precision:.4f}\nRecall: {self.recall:.4f}\nF1 Score: {self.f1_score:.4f}\nSHD: {self.shd:.4f}\n\n"
        scores_text_2 = f"TP: {self.TP}\nFP: {self.FP}\nTN: {self.TN}\nFN: {self.FN}\n\n"
        self.canvas_scores.delete(self.text_metric_scores)
        self.canvas_scores.create_text(150, 150, text=scores_text_1+scores_text_2, font=("Arial", 16), fill="black")


        # also print in terminal
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1 Score:", self.f1_score)
        print("SHD:", self.shd)
        print("\n")
        print("TP:", self.TP)
        print("FP:", self.FP)
        print("TN:", self.TN)
        print("FN:", self.FN)
        return self.accuracy, self.precision, self.recall, self.f1_score, self.shd, self.TP, self.FP, self.TN, self.FN
        
    def run(self):
        self.root.mainloop()

# Create the main window
root = tk.Tk()
app = DAGComparisonToolbox(root)
app.run()
