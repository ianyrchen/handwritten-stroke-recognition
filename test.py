import tkinter as tk
from tkinter import Menu
from tkinter import filedialog
import time
import pickle

class VirtualWhiteboard:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, bg="white", width=1000, height=800)
        self.canvas.pack()
        self.strokes = []  # List to store completed strokes
        self.current_stroke = []  # List to store the current stroke points

        self.canvas.bind("<Button-1>", self.start_recording)
        self.canvas.bind("<B1-Motion>", self.record_position)
        self.canvas.bind("<ButtonRelease-1>", self.stop_recording)
        self.root.bind("<Return>", self.delete_last_stroke)  # Bind Enter key to delete_last_stroke

        # Create a menu
        self.menu = Menu(root)
        root.config(menu=self.menu)
        
        # Add menu items
        file_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear_board)
        file_menu.add_command(label="Export", command=self.export_data)
        self.auto_export()
    def start_recording(self, event):
        """Start recording a new stroke."""
        self.current_stroke = []
        self.record_position(event)
    def record_position(self, event):
        #Record the cursor position and time."" 
        x, y = event.x, event.y
        t = time.time()
        # If this is not the first point in the stroke, draw a line to the current point
        if self.current_stroke:
            last_x, last_y, _ = self.current_stroke[-1]
            self.canvas.create_line(last_x, last_y, x, y, fill='black')
        self.current_stroke.append((x, y, t))
#        self.canvas.create_oval(x-1, y-1, x+1, y+1, fill='black')

    def stop_recording(self, event):
        # Stop recording the current stroke and save it.""
        if self.current_stroke:
            self.strokes.append([(50*x, 50*y, t) for x,y,t in self.current_stroke])
        self.current_stroke = []

    def clear_board(self):
        # ""Clear the canvas and reset strokes.""
        self.canvas.delete("all")
        self.strokes = []
    def export_data(self):
        """Export all stroke data to a pickle file."""
        #file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                       #          filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])A
        file_path = 'whiteboardtest.pkl'
        if True: #file_path:
            with open(file_path, 'wb') as f:
                pickle.dump([self.strokes], f)
            print(f"Data exported to {file_path}")

    def auto_export(self):
        """Automatically export data every second."""
        try:
            file_path = 'auto_export.pkl'
            self.export_data()
        except Exception as e:
            print(f"Failed to auto export data: {e}")
        
        # Schedule the next auto export in 1 second
        self.root.after(1000, self.auto_export)
    def delete_last_stroke(self, event):
        """Delete the last stroke from strokes and redraw the canvas."""
        if self.strokes:
            self.strokes.pop()
            self.redraw_canvas()
    
    def redraw_canvas(self):
        """Redraw the entire canvas."""
        self.canvas.delete("all")
        scale_factor = 50
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                x1, y1, _ = stroke[i-1]
                x2, y2, _ = stroke[i]
                self.canvas.create_line(x1 / scale_factor, y1 / scale_factor, x2 / scale_factor, y2 / scale_factor, fill='black')


# Create the main window and run the application
root = tk.Tk()
app = VirtualWhiteboard(root)
root.mainloop()
