"""tkinter GUI for Gerber Trace Mapper.

Entry point: gui_main()
Called when trace_mapper.py is run without file arguments or with --gui.

All matplotlib work is marshalled back to the main (tkinter) thread via
root.after() to avoid "main thread is not in main loop" errors.
"""

import matplotlib.pyplot as plt
from pathlib import Path

from process import (process_layers, collect_art_files,
                     load_cached_mappers, CacheMissError)
from plot import plot_comparison
from cache import CACHE_DIRNAME


class _LogWriter:
    """Redirect print() output to a GUI log callback."""
    def __init__(self, callback):
        self._cb = callback

    def write(self, text):
        if text:
            self._cb(text)

    def flush(self):
        pass


def gui_main():
    """Launch the tkinter GUI for the Gerber Trace Mapper."""
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    import threading
    import sys

    root = tk.Tk()
    root.title("Gerber Trace Mapper")
    root.geometry("720x620")
    root.resizable(True, True)

    # ---- File selection ----
    file_frame = ttk.LabelFrame(root, text="Gerber Files (.art / .gbr)")
    file_frame.pack(fill='x', padx=8, pady=(8, 4))

    file_listbox = tk.Listbox(file_frame, height=4, selectmode=tk.EXTENDED)
    file_listbox.pack(side='left', fill='both', expand=True, padx=(4, 0), pady=4)
    file_scroll = ttk.Scrollbar(file_frame, orient='vertical',
                                command=file_listbox.yview)
    file_scroll.pack(side='left', fill='y', pady=4)
    file_listbox.config(yscrollcommand=file_scroll.set)

    btn_frame = tk.Frame(file_frame)
    btn_frame.pack(side='left', padx=4, pady=4)

    def browse_files():
        paths = filedialog.askopenfilenames(
            title="Select Gerber files",
            filetypes=[("Gerber / Artwork", "*.art *.gbr"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in file_listbox.get(0, tk.END):
                file_listbox.insert(tk.END, p)

    def browse_dir():
        d = filedialog.askdirectory(title="Select directory containing Gerber files")
        if d:
            if d not in file_listbox.get(0, tk.END):
                file_listbox.insert(tk.END, d)

    def remove_selected():
        for idx in reversed(file_listbox.curselection()):
            file_listbox.delete(idx)

    ttk.Button(btn_frame, text="Add Files", command=browse_files).pack(fill='x', pady=1)
    ttk.Button(btn_frame, text="Add Dir", command=browse_dir).pack(fill='x', pady=1)
    ttk.Button(btn_frame, text="Remove", command=remove_selected).pack(fill='x', pady=1)

    # ---- Parameters ----
    param_frame = ttk.LabelFrame(root, text="Grid Parameters")
    param_frame.pack(fill='x', padx=8, pady=4)

    ttk.Label(param_frame, text="NX:").grid(row=0, column=0, padx=4, pady=2, sticky='e')
    nx_var = tk.StringVar(value="20")
    ttk.Entry(param_frame, textvariable=nx_var, width=8).grid(row=0, column=1, padx=4)

    ttk.Label(param_frame, text="NY:").grid(row=0, column=2, padx=4, pady=2, sticky='e')
    ny_var = tk.StringVar(value="20")
    ttk.Entry(param_frame, textvariable=ny_var, width=8).grid(row=0, column=3, padx=4)

    ttk.Label(param_frame, text="Merge Tolerance:").grid(
        row=0, column=4, padx=4, pady=2, sticky='e')
    tol_var = tk.StringVar(value="0.0")
    ttk.Entry(param_frame, textvariable=tol_var, width=10).grid(row=0, column=5, padx=4)

    ttk.Label(param_frame, text="Display Pixels:").grid(
        row=1, column=0, padx=4, pady=2, sticky='e')
    disp_var = tk.StringVar(value="600")
    ttk.Entry(param_frame, textvariable=disp_var, width=8).grid(row=1, column=1, padx=4)
    ttk.Label(param_frame, text="(left-panel raster, larger=sharper/slower)").grid(
        row=1, column=2, columnspan=4, padx=4, pady=2, sticky='w')

    # ---- Custom grid (non-uniform cell edges from CSV) ----
    custom_frame = ttk.LabelFrame(
        root, text="Custom Grid (optional: column-vector CSV of cell edges, "
                   "overrides NX/NY when both are set)")
    custom_frame.pack(fill='x', padx=8, pady=4)

    x_csv_var = tk.StringVar(value="")
    y_csv_var = tk.StringVar(value="")

    ttk.Label(custom_frame, text="X coords CSV:").grid(
        row=0, column=0, padx=4, pady=2, sticky='e')
    ttk.Entry(custom_frame, textvariable=x_csv_var).grid(
        row=0, column=1, padx=4, pady=2, sticky='ew')

    def browse_x_csv():
        p = filedialog.askopenfilename(
            title="Select X-coordinate CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            x_csv_var.set(p)

    ttk.Button(custom_frame, text="Browse", command=browse_x_csv).grid(
        row=0, column=2, padx=4, pady=2)
    ttk.Button(custom_frame, text="Clear",
               command=lambda: x_csv_var.set("")).grid(
        row=0, column=3, padx=4, pady=2)

    ttk.Label(custom_frame, text="Y coords CSV:").grid(
        row=1, column=0, padx=4, pady=2, sticky='e')
    ttk.Entry(custom_frame, textvariable=y_csv_var).grid(
        row=1, column=1, padx=4, pady=2, sticky='ew')

    def browse_y_csv():
        p = filedialog.askopenfilename(
            title="Select Y-coordinate CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            y_csv_var.set(p)

    ttk.Button(custom_frame, text="Browse", command=browse_y_csv).grid(
        row=1, column=2, padx=4, pady=2)
    ttk.Button(custom_frame, text="Clear",
               command=lambda: y_csv_var.set("")).grid(
        row=1, column=3, padx=4, pady=2)
    custom_frame.grid_columnconfigure(1, weight=1)

    # ---- Output directory ----
    out_frame = ttk.LabelFrame(root, text="Output Directory (blank = same as input)")
    out_frame.pack(fill='x', padx=8, pady=4)
    outdir_var = tk.StringVar(value="")
    ttk.Entry(out_frame, textvariable=outdir_var).pack(
        side='left', fill='x', expand=True, padx=4, pady=4)

    def browse_outdir():
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            outdir_var.set(d)

    ttk.Button(out_frame, text="Browse", command=browse_outdir).pack(
        side='left', padx=4, pady=4)

    # ---- Options ----
    opt_frame = ttk.LabelFrame(root, text="Options")
    opt_frame.pack(fill='x', padx=8, pady=4)

    polarity_var = tk.BooleanVar(value=True)
    even_odd_var = tk.BooleanVar(value=False)
    no_merge_var = tk.BooleanVar(value=False)
    interactive_var = tk.BooleanVar(value=False)
    shared_bounds_var = tk.BooleanVar(value=True)
    export_csv_var = tk.BooleanVar(value=True)
    plot_var = tk.BooleanVar(value=True)
    show_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(opt_frame, text="Use Gerber polarity (recommended)",
                    variable=polarity_var).grid(
        row=0, column=0, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Even-Odd fill (legacy)", variable=even_odd_var).grid(
        row=0, column=1, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="No Merge", variable=no_merge_var).grid(
        row=0, column=2, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Interactive exclude", variable=interactive_var).grid(
        row=0, column=3, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Shared bounds", variable=shared_bounds_var).grid(
        row=1, column=0, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Export CSV", variable=export_csv_var).grid(
        row=1, column=1, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Generate plots", variable=plot_var).grid(
        row=1, column=2, padx=6, pady=2, sticky='w')
    ttk.Checkbutton(opt_frame, text="Show plots", variable=show_var).grid(
        row=1, column=3, padx=6, pady=2, sticky='w')

    exclude_var = tk.BooleanVar(value=False)
    exclude_n_var = tk.StringVar(value="1")
    ttk.Checkbutton(opt_frame, text="Exclude largest poly", variable=exclude_var).grid(
        row=2, column=0, padx=6, pady=2, sticky='w')
    ef = tk.Frame(opt_frame)
    ef.grid(row=2, column=1, padx=6, pady=2, sticky='w')
    ttk.Label(ef, text="Count:").pack(side='left')
    ttk.Spinbox(ef, from_=1, to=20, textvariable=exclude_n_var,
                width=4).pack(side='left', padx=2)

    cache_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frame, text="Use raster cache", variable=cache_var).grid(
        row=2, column=2, padx=6, pady=2, sticky='w')

    # ---- Log output ----
    log_frame = ttk.LabelFrame(root, text="Log")
    log_frame.pack(fill='both', expand=True, padx=8, pady=4)
    log_text = tk.Text(log_frame, height=12, state='disabled', wrap='word')
    log_text.pack(fill='both', expand=True, padx=4, pady=4)
    log_scroll = ttk.Scrollbar(log_text, orient='vertical', command=log_text.yview)
    log_scroll.pack(side='right', fill='y')
    log_text.config(yscrollcommand=log_scroll.set)

    # log() is safe to call from any thread: marshals to the Tk main thread.
    def _append_log(msg: str):
        log_text.config(state='normal')
        log_text.insert(tk.END, msg)
        log_text.see(tk.END)
        log_text.config(state='disabled')

    def log(msg):
        try:
            root.after(0, lambda m=msg: _append_log(m))
        except RuntimeError:
            pass

    # ---- Run button ----
    run_frame = tk.Frame(root)
    run_frame.pack(fill='x', padx=8, pady=(0, 8))

    def run_processing():
        paths = list(file_listbox.get(0, tk.END))
        if not paths:
            messagebox.showwarning("No files", "Please add at least one Gerber file.")
            return

        try:
            nx = int(nx_var.get())
            ny = int(ny_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "NX and NY must be integers.")
            return
        try:
            merge_tol = float(tol_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Merge tolerance must be a number.")
            return
        try:
            disp_pix = max(50, int(disp_var.get()))
        except ValueError:
            messagebox.showerror("Invalid", "Display pixels must be an integer.")
            return

        outdir = outdir_var.get().strip() or None
        excl_n = 0
        if exclude_var.get():
            try:
                excl_n = int(exclude_n_var.get())
            except ValueError:
                excl_n = 1

        x_csv = x_csv_var.get().strip() or None
        y_csv = y_csv_var.get().strip() or None
        if (x_csv is None) != (y_csv is None):
            messagebox.showerror(
                "Custom grid",
                "Please provide BOTH the X and Y coordinate CSV files, "
                "or leave both empty to use NX/NY.")
            return

        # Snapshot all Tk vars on the main thread; Tk is not thread-safe.
        opts = {
            'shared_bounds': shared_bounds_var.get(),
            'export_csv': export_csv_var.get(),
            'no_merge': no_merge_var.get(),
            'interactive': interactive_var.get(),
            'even_odd': even_odd_var.get(),
            'use_polarity': polarity_var.get(),
            'cache': cache_var.get(),
        }
        do_plot = plot_var.get()
        do_show = show_var.get()
        run_btn.config(state='disabled')

        def worker():
            old_stdout = sys.stdout
            sys.stdout = _LogWriter(log)
            try:
                files = collect_art_files(paths)
                if not files:
                    log("No Gerber files found in the given paths.\n")
                    return
                log(f"Found {len(files)} Gerber file(s)\n")

                results = process_layers(
                    filepaths=files,
                    nx=nx, ny=ny,
                    shared_bounds=opts['shared_bounds'],
                    export_csv=opts['export_csv'],
                    plot=False,
                    show=False,
                    outdir=outdir,
                    merge_tolerance=merge_tol,
                    no_merge=opts['no_merge'],
                    interactive=opts['interactive'],
                    even_odd=opts['even_odd'],
                    use_polarity=opts['use_polarity'],
                    exclude_largest=excl_n,
                    min_display_pixels=disp_pix,
                    cache=opts['cache'],
                    x_coords_csv=x_csv,
                    y_coords_csv=y_csv,
                )

                log("\n=== Summary ===\n")
                for name, mapper in results.items():
                    info = mapper.grid_info
                    log(f"  {name}: {info['nx']}x{info['ny']} grid, "
                        f"avg Cu = {mapper.fractions.mean():.4f}, "
                        f"max = {mapper.fractions.max():.4f}\n")
                log("Done.\n")

                root.after(0, lambda: _finish_plots(results, files, outdir))

            except Exception as e:
                log(f"\nERROR: {e}\n")
                import traceback
                log(traceback.format_exc())
                root.after(0, lambda: run_btn.config(state='normal'))
            finally:
                sys.stdout = old_stdout

        def _finish_plots(results, files, out):
            """Generate plots on the main thread (matplotlib requirement)."""
            try:
                if not (do_plot or do_show):
                    return

                fp_by_name = {Path(fp).stem: fp for fp in files}

                for name, mapper in results.items():
                    fp = fp_by_name.get(name)
                    if fp is None:
                        continue
                    stub = type('LayerStub', (), {'name': name, 'filepath': fp})()
                    out_base = Path(out) if out else Path(fp).parent

                    if do_plot:
                        fig = plot_comparison(stub, mapper)
                        png_path = out_base / f"{name}.png"
                        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
                        if not do_show:
                            plt.close(fig)
                        log(f"Plot saved: {png_path}\n")

                if do_show:
                    plt.show()
            except Exception as e:
                log(f"\nPlot ERROR: {e}\n")
            finally:
                run_btn.config(state='normal')

        threading.Thread(target=worker, daemon=True).start()

    def show_saved_plot():
        """Display interactive plots from cached rasters without re-running.

        Requires a prior successful Run with matching parameters so the
        meta + raster caches exist. On any cache miss, a messagebox
        instructs the user to run once first.
        """
        paths = list(file_listbox.get(0, tk.END))
        if not paths:
            messagebox.showwarning("No files", "Please add at least one Gerber file.")
            return
        if not show_var.get():
            messagebox.showinfo(
                "Show plots disabled",
                "Enable 'Show plots' to display interactive figures.")
            return

        try:
            nx = int(nx_var.get())
            ny = int(ny_var.get())
            merge_tol = float(tol_var.get())
            disp_pix = max(50, int(disp_var.get()))
        except ValueError:
            messagebox.showerror(
                "Invalid", "NX, NY, merge tolerance, and display pixels "
                "must be valid numbers.")
            return

        excl_n = 0
        if exclude_var.get():
            try:
                excl_n = int(exclude_n_var.get())
            except ValueError:
                excl_n = 1

        x_csv = x_csv_var.get().strip() or None
        y_csv = y_csv_var.get().strip() or None
        if (x_csv is None) != (y_csv is None):
            messagebox.showerror(
                "Custom grid",
                "Please provide BOTH the X and Y coordinate CSV files, "
                "or leave both empty to use NX/NY.")
            return

        try:
            files = collect_art_files(paths)
            if not files:
                messagebox.showwarning(
                    "No files", "No .art / .gbr files found in the given paths.")
                return

            results = load_cached_mappers(
                filepaths=files,
                nx=nx, ny=ny,
                shared_bounds=shared_bounds_var.get(),
                merge_tolerance=merge_tol,
                no_merge=no_merge_var.get(),
                even_odd=even_odd_var.get(),
                use_polarity=polarity_var.get(),
                exclude_largest=excl_n,
                min_display_pixels=disp_pix,
                x_coords_csv=x_csv,
                y_coords_csv=y_csv,
            )
        except CacheMissError as e:
            messagebox.showerror(
                "No saved plot data",
                f"{e}\n\nParameters used here must match a previous Run "
                "(merge tolerance, polarity, even-odd, no-merge, exclude "
                "count, display pixels, shared bounds, custom grid).")
            return
        except Exception as e:
            messagebox.showerror("Error loading cache", str(e))
            log(f"\nERROR: {e}\n")
            return

        fp_by_name = {Path(fp).stem: fp for fp in files}
        for name, mapper in results.items():
            fp = fp_by_name.get(name)
            stub = type('LayerStub', (), {
                'name': name, 'filepath': fp or name})()
            plot_comparison(stub, mapper)
            log(f"Loaded saved plot: {name}\n")
        plt.show()

    def clear_cache():
        """Delete .trace_cache directories next to the listed art files."""
        import shutil
        roots = set()
        for p in file_listbox.get(0, tk.END):
            pp = Path(p)
            roots.add(pp if pp.is_dir() else pp.parent)
        removed = 0
        for r in roots:
            c = r / CACHE_DIRNAME
            if c.exists():
                try:
                    shutil.rmtree(c)
                    removed += 1
                except Exception as e:
                    log(f"  Failed to remove {c}: {e}\n")
        log(f"Cleared cache in {removed} location(s)\n")

    run_btn = ttk.Button(run_frame, text="Run", command=run_processing)
    run_btn.pack(side='left', padx=4)
    ttk.Button(run_frame, text="Show Saved Plot",
               command=show_saved_plot).pack(side='left', padx=4)
    ttk.Button(run_frame, text="Clear Cache", command=clear_cache).pack(
        side='left', padx=4)
    ttk.Button(run_frame, text="Quit", command=root.destroy).pack(side='right', padx=4)

    root.mainloop()
