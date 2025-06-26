import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import webbrowser
import subprocess
import sys
import platform
from pathlib import Path
import threading
from typing import List, Tuple, Optional
import os
import markdown

# Import your existing search engine classes
from Search_engine import AcademicSearchEngine, Document
# For now, I'll include the necessary parts inline

class SearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Search Engine")
        self.root.geometry("1200x800")
        
        # Initialize search engine
        self.engine = None
        self.current_results = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Folder selection
        ttk.Label(main_frame, text="Document Folder:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=60)
        self.folder_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=1)
        ttk.Button(folder_frame, text="Index", command=self.index_documents).grid(row=0, column=2, padx=(5, 0))
        
        ttk.Button(folder_frame, text="Rebuild", command=lambda: self.index_documents(force_rebuild=True)).grid(row=0, column=3, padx=(5, 0))


        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Search section
        search_frame = ttk.LabelFrame(main_frame, text="Search", padding="5")
        search_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(0, weight=1)
        
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(search_frame, textvariable=self.query_var, font=("Arial", 12))
        query_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        query_entry.bind('<Return>', lambda event: self.search_documents())
        
        search_button = ttk.Button(search_frame, text="Search", command=self.search_documents)
        search_button.grid(row=0, column=1)
        
        # Search options
        options_frame = ttk.Frame(search_frame)
        options_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(options_frame, text="Results:").grid(row=0, column=0, padx=(0, 5))
        self.top_k_var = tk.StringVar(value="10")
        top_k_combo = ttk.Combobox(options_frame, textvariable=self.top_k_var, values=["5", "10", "15", "20", "30"], width=5)
        top_k_combo.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(options_frame, text="Min Score:").grid(row=0, column=2, padx=(0, 5))
        self.min_score_var = tk.StringVar(value="0.2")
        min_score_entry = ttk.Entry(options_frame, textvariable=self.min_score_var, width=8)
        min_score_entry.grid(row=0, column=3)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Search Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create treeview for results
        columns = ("Score", "File", "Page", "Type")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="tree headings", height=8)
        
        # Configure columns
        self.results_tree.heading("#0", text="Rank")
        self.results_tree.column("#0", width=50, minwidth=50)
        
        self.results_tree.heading("Score", text="Score")
        self.results_tree.column("Score", width=80, minwidth=80)
        
        self.results_tree.heading("File", text="File")
        self.results_tree.column("File", width=300, minwidth=200)
        
        self.results_tree.heading("Page", text="Page")
        self.results_tree.column("Page", width=60, minwidth=60)
        
        self.results_tree.heading("Type", text="Type")
        self.results_tree.column("Type", width=80, minwidth=80)
        
        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll_x = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Content preview
        preview_frame = ttk.LabelFrame(main_frame, text="Content Preview", padding="5")
        preview_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        self.content_text = scrolledtext.ScrolledText(preview_frame, height=10, wrap=tk.WORD)
        self.content_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        buttons_frame = ttk.Frame(preview_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.open_file_button = ttk.Button(buttons_frame, text="Open File", command=self.open_selected_file, state="disabled")
        self.open_file_button.grid(row=0, column=0, padx=(0, 5))
        
        self.copy_content_button = ttk.Button(buttons_frame, text="Copy Content", command=self.copy_content, state="disabled")
        self.copy_content_button.grid(row=0, column=1)
        
        # Bind treeview selection
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)
        self.results_tree.bind("<Double-1>", self.on_result_double_click)
        
        # Configure grid weights for main window
        main_frame.rowconfigure(4, weight=1)
        
    def browse_folder(self):
        """Browse for document folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)
    
    def index_documents(self, force_rebuild=False):
        """Index documents in the selected folder"""
        folder_path = self.folder_var.get().strip()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder first")
            return
        
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", "Selected folder does not exist")
            return
        
        # Start indexing in a separate thread
        self.status_var.set("Indexing documents...")
        self.root.config(cursor="wait")
        
        def index_thread():
            # try:
                # Import here to avoid issues if not available
                from Search_engine import AcademicSearchEngine
                
                self.engine = AcademicSearchEngine()
                self.engine.index_folder(folder_path, force_rebuild=force_rebuild)

                
                # Update UI in main thread
                self.root.after(0, self.indexing_complete)
                
            # except ImportError:
            #     self.root.after(0, lambda: self.show_error("Please ensure your search engine module is available"))
            # except Exception as e:
            #     self.root.after(0, lambda: self.show_error(f"Indexing failed: {str(e)}"))
        
        threading.Thread(target=index_thread, daemon=True).start()
    
    def indexing_complete(self):
        """Called when indexing is complete"""
        self.status_var.set("Ready - Documents indexed successfully")
        self.root.config(cursor="")
        messagebox.showinfo("Success", "Documents indexed successfully!")
    
    def show_error(self, message):
        """Show error message"""
        self.status_var.set("Error occurred")
        self.root.config(cursor="")
        messagebox.showerror("Error", message)
    
    def search_documents(self):
        """Search for documents"""
        if self.engine is None:
            messagebox.showerror("Error", "Please index documents first")
            return
        
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query")
            return
        
        try:
            top_k = int(self.top_k_var.get())
            min_score = float(self.min_score_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid search parameters")
            return
        
        # Clear previous results
        self.clear_results()
        self.status_var.set("Searching...")
        
        def search_thread():
            try:
                results = self.engine.search(query, top_k=top_k, min_score=min_score)
                self.root.after(0, lambda: self.display_results(results))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Search failed: {str(e)}"))
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def display_results(self, results):
        """Display search results in the treeview"""
        self.current_results = results
        
        # Clear previous results
        self.clear_results()
        
        if not results:
            self.status_var.set("No results found")
            return
        
        # Populate treeview
        for i, (doc, score) in enumerate(results, 1):
            file_name = Path(doc.file_path).name
            page_str = str(doc.page_num) if doc.page_num else ""
            
            item_id = self.results_tree.insert("", "end", 
                                             text=str(i),
                                             values=(f"{score:.3f}", file_name, page_str, doc.file_type))
        
        self.status_var.set(f"Found {len(results)} results")
    
    def clear_results(self):
        """Clear all results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.content_text.delete(1.0, tk.END)
        self.open_file_button.config(state="disabled")
        self.copy_content_button.config(state="disabled")
    
    def on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        rank = int(self.results_tree.item(item, "text")) - 1
        
        if 0 <= rank < len(self.current_results):
            doc, score = self.current_results[rank]
            
            # Display content
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(1.0, doc.text)
            
            # Enable buttons
            self.open_file_button.config(state="normal")
            self.copy_content_button.config(state="normal")
    
    def on_result_double_click(self, event):
        """Handle double-click on result"""
        self.open_selected_file()
    
    def open_selected_file(self):
        """Open the selected file"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        rank = int(self.results_tree.item(item, "text")) - 1
        
        if 0 <= rank < len(self.current_results):
            doc, score = self.current_results[rank]
            self.open_file(doc.file_path, doc.page_num)

    def open_markdown_file(self, file_path: str, page_num: Optional[int] = None):
        # Convert Markdown to HTML
        # Read Markdown content
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert Markdown to HTML (with MathJax support)
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'pymdownx.arithmatex',  # LaTeX math support
                'pymdownx.superfences',  # Needed for code blocks
                'pymdownx.highlight'    # Syntax highlighting
            ],
            extension_configs={
                'pymdownx.arithmatex': {
                    'generic': True,  # Use MathJax
                }
            }
        )

        html_file = "temp_rendered.html"
        # Check if the file exists, if not create it
        if not os.path.exists(html_file):
            with open(html_file, "w", encoding="utf-8") as f:
                f.write("")

        # Write HTML with MathJax loading
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Markdown Preview</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/sindresorhus/github-markdown-css@5/github-markdown.min.css">
                <style>
                    .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}
                    @media (max-width: 767px) {{ .markdown-body {{ padding: 15px; }} }}
                </style>
                <!-- Load MathJax -->
                <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            </head>
            <body class="markdown-body">
                {html_content}
            </body>
            </html>
            """)

    
    def open_file(self, file_path: str, page_num: Optional[int] = None):
        """Open file with system default application"""
        try:
            file_path = Path(file_path).resolve()
            
            print(page_num)

            if page_num is not None:
                # Format the path with the page number
                formatted_path = f"file:///{file_path.as_posix()}#page={page_num}"
            else:
                formatted_path = f"file:///{file_path.as_posix()}"
            
            print(f"Opening: {formatted_path}")  # Debugging output

            if file_path.suffix.lower() == '.md':
                # If it's a Markdown file, open it in a web browser
                self.open_markdown_file(file_path, page_num)
                webbrowser.open(f"file:///{Path('temp_rendered.html').resolve().as_posix()}")
                return


            if not file_path.exists():
                messagebox.showerror("Error", f"File not found: {file_path}")
                return
            
            system = platform.system()
            
            if system == "Windows":
                # os.startfile(str(formatted_path))
                subprocess.run([r"C:\Program Files\Mozilla Firefox\firefox.exe", formatted_path])
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(formatted_path)])
            else:  # Linux and others
                subprocess.run(["xdg-open", str(formatted_path)])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    def copy_content(self):
        """Copy selected content to clipboard"""
        content = self.content_text.get(1.0, tk.END).strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("Content copied to clipboard")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SearchGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()