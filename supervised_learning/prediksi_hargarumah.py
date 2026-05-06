import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── Default dataset ───────────────────────────────────────────────────────────
DEFAULT_X = [50, 65, 80, 90, 100, 110, 120, 135, 150, 160, 175, 190, 200, 220, 240]
DEFAULT_Y = [120, 150, 175, 200, 220, 245, 260, 295, 320, 345, 370, 410, 430, 465, 500]

# ── Color palette ─────────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
CARD    = "#21253a"
ACCENT  = "#6c8fff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#4caf91"
ACCENT4 = "#f5a623"
TEXT    = "#e8eaf6"
MUTED   = "#7c82a8"
SUCCESS = "#4caf91"
BORDER  = "#2e3250"


class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eksplorasi Regresi Linear")
        self.root.configure(bg=BG)
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.x_data = list(DEFAULT_X)
        self.y_data = list(DEFAULT_Y)
        self.model  = None

        self._build_ui()
        self._run_regression()

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=BG, pady=12)
        title_frame.pack(fill="x", padx=20)
        tk.Label(title_frame, text="Eksplorasi Regresi Linear",
                 font=("Georgia", 18, "bold"), bg=BG, fg=TEXT).pack(side="left")
        tk.Label(title_frame, text="ŷ = wx + b",
                 font=("Courier", 14), bg=BG, fg=ACCENT).pack(side="right")

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        left = tk.Frame(content, bg=BG, width=300)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)
        self._build_controls(left)

        right = tk.Frame(content, bg=SURFACE, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="left", fill="both", expand=True)
        self._build_tabs(right)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        outer.pack(fill="x", pady=(0, 10))
        tk.Label(outer, text=title, font=("Georgia", 11, "bold"),
                 bg=CARD, fg=ACCENT, pady=8, padx=12).pack(anchor="w")
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x")
        inner = tk.Frame(outer, bg=CARD, padx=12, pady=10)
        inner.pack(fill="x")
        return inner

    def _metric_row(self, parent, label, var, color=None):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=("Courier", 10), bg=CARD,
                 fg=MUTED, width=14, anchor="w").pack(side="left")
        tk.Label(row, textvariable=var, font=("Courier", 11, "bold"),
                 bg=CARD, fg=color or SUCCESS).pack(side="left")

    # ── Controls ──────────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        metrics = self._card(parent, "Metrik Model")
        self.var_slope     = tk.StringVar(value="—")
        self.var_intercept = tk.StringVar(value="—")
        self.var_r2        = tk.StringVar(value="—")
        self.var_rmse      = tk.StringVar(value="—")
        self._metric_row(metrics, "Slope  (w)", self.var_slope)
        self._metric_row(metrics, "Intercept (b)", self.var_intercept)
        self._metric_row(metrics, "R² score", self.var_r2)
        self._metric_row(metrics, "RMSE", self.var_rmse)

        loss_card = self._card(parent, "Loss functions")
        self.var_mse   = tk.StringVar(value="—")
        self.var_mae   = tk.StringVar(value="—")
        self.var_rmse2 = tk.StringVar(value="—")
        self._metric_row(loss_card, "MSE", self.var_mse,   color=ACCENT2)
        self._metric_row(loss_card, "MAE", self.var_mae,   color=ACCENT4)
        self._metric_row(loss_card, "RMSE", self.var_rmse2, color=ACCENT3)
        tk.Label(loss_card, text="MSE  = (1/n)·Σ(y−ŷ)²",
                 font=("Courier", 9), bg=CARD, fg=MUTED).pack(anchor="w", pady=(6,0))
        tk.Label(loss_card, text="MAE  = (1/n)·Σ|y−ŷ|",
                 font=("Courier", 9), bg=CARD, fg=MUTED).pack(anchor="w")
        tk.Label(loss_card, text="RMSE = √MSE",
                 font=("Courier", 9), bg=CARD, fg=MUTED).pack(anchor="w")

        pred = self._card(parent, "Prediksi Nilai")
        tk.Label(pred, text="Input x:", font=("Courier", 10),
                 bg=CARD, fg=MUTED).pack(anchor="w")
        self.pred_entry = tk.Entry(pred, font=("Courier", 12), bg=SURFACE,
                                   fg=TEXT, insertbackground=TEXT,
                                   relief="flat", bd=4)
        self.pred_entry.pack(fill="x", pady=(4, 8))
        self.pred_entry.insert(0, "130")
        tk.Button(pred, text="Predict  →", font=("Georgia", 11),
                  bg=ACCENT, fg="white", relief="flat", bd=0,
                  cursor="hand2", pady=6,
                  command=self._predict).pack(fill="x")
        self.pred_result = tk.Label(pred, text="", font=("Courier", 13, "bold"),
                                    bg=CARD, fg=ACCENT2, pady=6)
        self.pred_result.pack()

        add = self._card(parent, "Tambah Data")
        row = tk.Frame(add, bg=CARD)
        row.pack(fill="x", pady=(0, 6))
        tk.Label(row, text="x:", font=("Courier", 10), bg=CARD,
                 fg=MUTED, width=3).pack(side="left")
        self.new_x = tk.Entry(row, font=("Courier", 11), bg=SURFACE,
                               fg=TEXT, insertbackground=TEXT,
                               relief="groove", bd=2, width=10,
                               highlightthickness=1, highlightbackground=BORDER)
        self.new_x.pack(side="left", padx=(0, 10))
        tk.Label(row, text="y:", font=("Courier", 10), bg=CARD,
                 fg=MUTED, width=3).pack(side="left")
        self.new_y = tk.Entry(row, font=("Courier", 11), bg=SURFACE,
                               fg=TEXT, insertbackground=TEXT,
                               relief="groove", bd=2, width=10,
                               highlightthickness=1, highlightbackground=BORDER)
        self.new_y.pack(side="left")
        tk.Button(add, text="Add point  +", font=("Georgia", 10),
                  bg=SURFACE, fg=ACCENT, relief="flat", bd=0,
                  highlightthickness=1, highlightbackground=ACCENT,
                  cursor="hand2", pady=5,
                  command=self._add_point).pack(fill="x", pady=(4, 0))

        self.show_residuals = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="  Show residuals",
                       variable=self.show_residuals,
                       font=("Georgia", 10), bg=BG, fg=MUTED,
                       activebackground=BG, activeforeground=TEXT,
                       selectcolor=SURFACE, cursor="hand2",
                       command=self._run_regression).pack(anchor="w", pady=(10, 0))

        tk.Button(parent, text="Reset to default data",
                  font=("Georgia", 10), bg=SURFACE, fg=MUTED,
                  relief="flat", bd=0, cursor="hand2", pady=6,
                  command=self._reset).pack(fill="x", pady=(6, 0))

        self.var_n = tk.StringVar()
        tk.Label(parent, textvariable=self.var_n, font=("Courier", 9),
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(4, 0))

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _build_tabs(self, parent):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=SURFACE, borderwidth=0, tabmargins=0)
        style.configure("Dark.TNotebook.Tab",
                        background=CARD, foreground=MUTED,
                        font=("Georgia", 10), padding=[14, 6], borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", SURFACE)],
                  foreground=[("selected", TEXT)])

        nb = ttk.Notebook(parent, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)

        tab1 = tk.Frame(nb, bg=SURFACE)
        nb.add(tab1, text="  Grafik Regresi  ")
        self._build_fit_chart(tab1)

        tab2 = tk.Frame(nb, bg=SURFACE)
        nb.add(tab2, text="  Loss function  ")
        self._build_loss_chart(tab2)

        tab3 = tk.Frame(nb, bg=SURFACE)
        nb.add(tab3, text="  Gradient descent  ")
        self._build_gd_tab(tab3)

    # ── Fit chart ─────────────────────────────────────────────────────────────
    def _build_fit_chart(self, parent):
        self.fig1 = Figure(figsize=(6, 5), dpi=100, facecolor=SURFACE)
        self.ax1  = self.fig1.add_subplot(111)
        self.fig1.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.1)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=parent)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

    def _draw_fit_chart(self, x_arr, y_arr, y_line, x_line):
        ax = self.ax1
        ax.clear()
        ax.set_facecolor(SURFACE)
        self.fig1.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("x (input)", color=MUTED, fontsize=10)
        ax.set_ylabel("y (output)", color=MUTED, fontsize=10)
        ax.set_title("Regression fit", color=TEXT, fontsize=12, pad=10)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.plot(x_line, y_line, color=ACCENT, linewidth=2,
                label="Regression line", zorder=2)
        if self.show_residuals.get() and self.model:
            y_hat = self.model.predict(x_arr.reshape(-1, 1)).flatten()
            for xi, yi, yhi in zip(x_arr, y_arr, y_hat):
                ax.plot([xi, xi], [yi, yhi], color=ACCENT2,
                        linewidth=1, alpha=0.6, zorder=1)
        ax.scatter(x_arr, y_arr, color=ACCENT2, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, label="Data points")
        try:
            px = float(self.pred_entry.get())
            if self.model:
                py = self.model.predict([[px]])[0]
                ax.scatter([px], [py], color="#ffe066", s=120, zorder=5,
                           marker="*", label=f"Prediction ({px:.0f}, {py:.1f})")
        except ValueError:
            pass
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        self.canvas1.draw()

    # ── Loss chart ────────────────────────────────────────────────────────────
    def _build_loss_chart(self, parent):
        self.fig2 = Figure(figsize=(6, 5), dpi=100, facecolor=SURFACE)
        self.fig2.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12,
                                  hspace=0.5)
        self.ax2a = self.fig2.add_subplot(211)
        self.ax2b = self.fig2.add_subplot(212)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=parent)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

    def _draw_loss_chart(self, x_arr, y_arr):
        if not self.model:
            return
        y_hat     = self.model.predict(x_arr.reshape(-1, 1)).flatten()
        residuals = y_arr - y_hat
        sq_errors = residuals ** 2
        abs_errors = np.abs(residuals)

        # Top: squared errors bar chart
        ax = self.ax2a
        ax.clear()
        ax.set_facecolor(SURFACE)
        self.fig2.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title("Squared error per point  (y − ŷ)²",
                     color=TEXT, fontsize=10, pad=6)
        ax.set_xlabel("data point index", color=MUTED, fontsize=9)
        ax.set_ylabel("(y − ŷ)²", color=MUTED, fontsize=9)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.4, alpha=0.6, axis="y")
        colors = [ACCENT2 if r > 0 else ACCENT for r in residuals]
        ax.bar(range(len(sq_errors)), sq_errors, color=colors, width=0.6, alpha=0.85)
        mse_val = np.mean(sq_errors)
        ax.axhline(mse_val, color=ACCENT4, linewidth=1.5, linestyle="--",
                   label=f"MSE = {mse_val:.1f}")
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        # Bottom: loss comparison
        ax2 = self.ax2b
        ax2.clear()
        ax2.set_facecolor(SURFACE)
        for spine in ax2.spines.values():
            spine.set_edgecolor(BORDER)
        ax2.tick_params(colors=MUTED, labelsize=9)
        ax2.set_title("MSE vs MAE vs RMSE", color=TEXT, fontsize=10, pad=6)
        ax2.grid(True, color=BORDER, linestyle="--", linewidth=0.4, alpha=0.6, axis="x")
        mse  = np.mean(sq_errors)
        mae  = np.mean(abs_errors)
        rmse = np.sqrt(mse)
        labels     = ["MSE\n(1/n)·Σ(y−ŷ)²", "MAE\n(1/n)·Σ|y−ŷ|", "RMSE\n√MSE"]
        values     = [mse, mae, rmse]
        bar_colors = [ACCENT2, ACCENT4, ACCENT3]
        bars = ax2.barh(labels, values, color=bar_colors, height=0.45, alpha=0.85)
        for bar, val in zip(bars, values):
            ax2.text(val + max(values) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", color=TEXT, fontsize=9)
        ax2.set_xlim(0, max(values) * 1.25)
        self.canvas2.draw()

    # ── Gradient descent tab ──────────────────────────────────────────────────
    def _build_gd_tab(self, parent):
        ctrl = tk.Frame(parent, bg=SURFACE, pady=8)
        ctrl.pack(fill="x", padx=14)

        tk.Label(ctrl, text="Learning rate (α):", font=("Courier", 10),
                 bg=SURFACE, fg=MUTED).pack(side="left")
        self.lr_var = tk.DoubleVar(value=0.0001)
        tk.Entry(ctrl, textvariable=self.lr_var, font=("Courier", 10),
                 bg=CARD, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=4, width=10).pack(side="left", padx=8)

        tk.Label(ctrl, text="Iterations:", font=("Courier", 10),
                 bg=SURFACE, fg=MUTED).pack(side="left")
        self.iters_var = tk.IntVar(value=200)
        tk.Entry(ctrl, textvariable=self.iters_var, font=("Courier", 10),
                 bg=CARD, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=4, width=8).pack(side="left", padx=8)

        tk.Button(ctrl, text="Run gradient descent  →",
                  font=("Georgia", 10), bg=ACCENT, fg="white",
                  relief="flat", bd=0, cursor="hand2", pady=4,
                  command=self._run_gd).pack(side="left", padx=12)

        self.gd_status = tk.Label(ctrl, text="", font=("Courier", 9),
                                  bg=SURFACE, fg=MUTED)
        self.gd_status.pack(side="left")

        self.fig3 = Figure(figsize=(6, 5), dpi=100, facecolor=SURFACE)
        self.fig3.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12,
                                  hspace=0.5)
        self.ax3a = self.fig3.add_subplot(211)
        self.ax3b = self.fig3.add_subplot(212)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=parent)
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)
        self._draw_gd_placeholder()

    def _draw_gd_placeholder(self):
        for ax, title in [(self.ax3a, "MSE loss curve  (press Run to start)"),
                          (self.ax3b, "w and b convergence")]:
            ax.clear()
            ax.set_facecolor(SURFACE)
            self.fig3.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.tick_params(colors=MUTED)
            ax.set_title(title, color=MUTED, fontsize=10)
        self.canvas3.draw()

    def _run_gd(self):
        try:
            lr    = float(self.lr_var.get())
            iters = int(self.iters_var.get())
        except Exception:
            messagebox.showerror("Input error", "Invalid learning rate or iterations.")
            return
        if lr <= 0 or iters <= 0:
            messagebox.showerror("Input error", "Values must be > 0.")
            return

        x_arr = np.array(self.x_data, dtype=float)
        y_arr = np.array(self.y_data, dtype=float)
        n = len(x_arr)

        # Normalize x for numerical stability
        x_mean = x_arr.mean()
        x_std  = x_arr.std() or 1.0
        x_norm = (x_arr - x_mean) / x_std

        w, b = 0.0, 0.0
        mse_history, w_history, b_history = [], [], []

        for _ in range(iters):
            y_pred = w * x_norm + b
            errors = y_pred - y_arr
            dw = (2 / n) * np.dot(errors, x_norm)
            db = (2 / n) * np.sum(errors)
            w -= lr * dw
            b -= lr * db
            mse_history.append(np.mean(errors ** 2))
            w_history.append(w)
            b_history.append(b)

        self.gd_status.config(
            text=f"Final  MSE: {mse_history[-1]:.2f}   w: {w:.4f}   b: {b:.2f}")
        self._draw_gd_charts(mse_history, w_history, b_history)

    def _draw_gd_charts(self, mse_hist, w_hist, b_hist):
        iters = range(1, len(mse_hist) + 1)

        ax = self.ax3a
        ax.clear()
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title("MSE loss curve — model learning over iterations",
                     color=TEXT, fontsize=10, pad=6)
        ax.set_xlabel("iteration", color=MUTED, fontsize=9)
        ax.set_ylabel("MSE loss", color=MUTED, fontsize=9)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.4, alpha=0.6)
        ax.plot(iters, mse_hist, color=ACCENT2, linewidth=1.8, label="MSE")
        ax.axhline(mse_hist[-1], color=ACCENT4, linewidth=1, linestyle="--",
                   label=f"Final MSE: {mse_hist[-1]:.1f}")
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        ax2 = self.ax3b
        ax2.clear()
        ax2.set_facecolor(SURFACE)
        for spine in ax2.spines.values():
            spine.set_edgecolor(BORDER)
        ax2.tick_params(colors=MUTED, labelsize=8)
        ax2.set_title("Weight (w) and bias (b) converging",
                      color=TEXT, fontsize=10, pad=6)
        ax2.set_xlabel("iteration", color=MUTED, fontsize=9)
        ax2.grid(True, color=BORDER, linestyle="--", linewidth=0.4, alpha=0.6)
        ax2.plot(iters, w_hist, color=ACCENT,  linewidth=1.8, label="w (slope)")
        ax2.plot(iters, b_hist, color=ACCENT3, linewidth=1.8, label="b (intercept)")
        ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        self.canvas3.draw()

    # ── Regression engine ─────────────────────────────────────────────────────
    def _run_regression(self):
        if len(self.x_data) < 2:
            return
        x_arr = np.array(self.x_data)
        y_arr = np.array(self.y_data)
        self.model = LinearRegression()
        self.model.fit(x_arr.reshape(-1, 1), y_arr)
        w      = self.model.coef_[0]
        b      = self.model.intercept_
        y_pred = self.model.predict(x_arr.reshape(-1, 1))
        r2     = r2_score(y_arr, y_pred)
        mse    = mean_squared_error(y_arr, y_pred)
        mae    = mean_absolute_error(y_arr, y_pred)
        rmse   = np.sqrt(mse)
        self.var_slope.set(f"{w:.4f}")
        self.var_intercept.set(f"{b:.2f}")
        self.var_r2.set(f"{r2:.4f}")
        self.var_rmse.set(f"{rmse:.2f}")
        self.var_mse.set(f"{mse:.2f}")
        self.var_mae.set(f"{mae:.2f}")
        self.var_rmse2.set(f"{rmse:.2f}")
        self.var_n.set(f"{len(self.x_data)} data points")
        x_line = np.linspace(min(x_arr) - 10, max(x_arr) + 10, 200)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        self._draw_fit_chart(x_arr, y_arr, y_line, x_line)
        self._draw_loss_chart(x_arr, y_arr)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _predict(self):
        if not self.model:
            return
        try:
            px = float(self.pred_entry.get())
            py = self.model.predict([[px]])[0]
            self.pred_result.config(text=f"ŷ = {py:.2f}")
            self._run_regression()
        except ValueError:
            self.pred_result.config(text="")
            messagebox.showerror("Input error", "Please enter a valid number for x.")

    def _add_point(self):
        try:
            nx = float(self.new_x.get())
            ny = float(self.new_y.get())
            self.x_data.append(nx)
            self.y_data.append(ny)
            self.new_x.delete(0, "end")
            self.new_y.delete(0, "end")
            self._run_regression()
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numbers for x and y.")

    def _reset(self):
        self.x_data = list(DEFAULT_X)
        self.y_data = list(DEFAULT_Y)
        self.pred_result.config(text="")
        self._run_regression()
        self._draw_gd_placeholder()
        self.gd_status.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app  = LinearRegressionApp(root)
    root.mainloop()