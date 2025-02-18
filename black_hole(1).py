import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================================================
# Global Parameters
# =========================================================
G = 1.0      # Gravitational constant (scaled)
DT = 0.01    # Time step (Euler integration)
SIZE = 10    # Half-size of the viewing area => [-SIZE..SIZE]
REMOVE_THRESHOLD = 0.5  # Distance for "Remove Object" mode

# --- Merge/Swallow Radii ---
BH_BH_MERGE_RADIUS = 0.5        # BH-BH merge distance
PARTICLE_MERGE_RADIUS = 0.2     # Particle-Particle merge distance
BH_PARTICLE_SWALLOW_RADIUS = 0.2# If particle is within this distance of BH => swallowed

# Default masses
DEFAULT_BH_MASS = 50.0
DEFAULT_PARTICLE_MASS = 1.0

# ---------------------------------------------------------
# Data Structures
# Each black_hole is [mass, [x,y], [vx,vy]]
# Each particle  is [mass, [x,y], [vx,vy]]
# ---------------------------------------------------------
black_holes = []
particles = []

# =========================================================
# Helper / Physics Functions
# =========================================================

def distance(pos1, pos2):
    """Euclidean distance between 2D points [x1,y1], [x2,y2]."""
    return np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

def merge_objects(m1, pos1, vel1, m2, pos2, vel2):
    """
    Merge two objects (BH or particle) with masses m1, m2, etc.
    Returns [m_new, [x_new, y_new], [vx_new, vy_new]].
    Momentum-conserving merge: new position = center of mass,
    new velocity = total momentum / total mass.
    """
    m_new = m1 + m2
    # Center of mass
    x_new = (m1*pos1[0] + m2*pos2[0]) / m_new
    y_new = (m1*pos1[1] + m2*pos2[1]) / m_new
    # Momentum
    vx_new = (m1*vel1[0] + m2*vel2[0]) / m_new
    vy_new = (m1*vel1[1] + m2*vel2[1]) / m_new
    return [m_new, [x_new, y_new], [vx_new, vy_new]]

def compute_acceleration_on_object(index, is_bh):
    """
    Compute gravitational acceleration on either:
      - black_holes[index], if is_bh=True
      - particles[index],    if is_bh=False
    from ALL objects (both BHs + particles).
    """
    if is_bh:
        m_i, pos_i, vel_i = black_holes[index]
    else:
        m_i, pos_i, vel_i = particles[index]
    
    ax, ay = 0.0, 0.0
    x_i, y_i = pos_i

    # -- BHs --
    for j, (m_bh, pos_bh, vel_bh) in enumerate(black_holes):
        # skip self if is_bh=True and j==index
        if is_bh and j == index:
            continue
        dx = x_i - pos_bh[0]
        dy = y_i - pos_bh[1]
        r2 = dx*dx + dy*dy
        r = np.sqrt(r2) + 1e-10
        a_mag = G * m_bh / r2
        ax += -a_mag * (dx / r)
        ay += -a_mag * (dy / r)

    # -- Particles --
    for j, (m_p, pos_p, vel_p) in enumerate(particles):
        # skip self if is_bh=False and j==index
        if not is_bh and j == index:
            continue
        dx = x_i - pos_p[0]
        dy = y_i - pos_p[1]
        r2 = dx*dx + dy*dy
        r = np.sqrt(r2) + 1e-10
        a_mag = G * m_p / r2
        ax += -a_mag * (dx / r)
        ay += -a_mag * (dy / r)

    return np.array([ax, ay])

# =========================================================
# Merging / Swallowing Logic
# =========================================================
def merge_bh_bh():
    """
    Merge black holes if within BH_BH_MERGE_RADIUS.
    We do multiple passes until no merges remain in this frame.
    """
    merged_any = True
    while merged_any:
        merged_any = False
        n = len(black_holes)
        if n < 2:
            break
        found_pair = False
        for i in range(n):
            if found_pair: 
                break
            for j in range(i+1, n):
                dist = distance(black_holes[i][1], black_holes[j][1])
                if dist < BH_BH_MERGE_RADIUS:
                    # Merge them
                    m1, pos1, vel1 = black_holes[i]
                    m2, pos2, vel2 = black_holes[j]
                    new_bh = merge_objects(m1, pos1, vel1, m2, pos2, vel2)
                    # Remove i, j
                    if j>i:
                        black_holes.pop(j)
                        black_holes.pop(i)
                    else:
                        black_holes.pop(i)
                        black_holes.pop(j)
                    # Insert new BH
                    black_holes.append(new_bh)
                    merged_any = True
                    found_pair = True
                    break

def merge_particle_particle():
    """
    Merge particles if within PARTICLE_MERGE_RADIUS.
    Repeat until no merges remain.
    """
    merged_any = True
    while merged_any:
        merged_any = False
        n = len(particles)
        if n < 2:
            break
        found_pair = False
        for i in range(n):
            if found_pair:
                break
            for j in range(i+1, n):
                dist = distance(particles[i][1], particles[j][1])
                if dist < PARTICLE_MERGE_RADIUS:
                    # Merge them
                    m1, pos1, vel1 = particles[i]
                    m2, pos2, vel2 = particles[j]
                    new_p = merge_objects(m1, pos1, vel1, m2, pos2, vel2)
                    # remove i,j
                    if j>i:
                        particles.pop(j)
                        particles.pop(i)
                    else:
                        particles.pop(i)
                        particles.pop(j)
                    # insert new
                    particles.append(new_p)
                    merged_any = True
                    found_pair = True
                    break

def bh_swallow_particles():
    """
    If a particle is within BH_PARTICLE_SWALLOW_RADIUS of a black hole,
    the BH's mass increases by the particle's mass, and the particle is removed.
    We'll do multiple passes until no more swallows happen.
    (If many BHs can swallow the same particle simultaneously, the first we find gets it.)
    """
    any_swallowed = True
    while any_swallowed:
        any_swallowed = False
        if not particles or not black_holes:
            break
        # We'll check each particle vs each BH
        found_swallow = False
        for p_index, (m_p, pos_p, vel_p) in enumerate(particles):
            px, py = pos_p
            for b_index, (m_bh, pos_bh, vel_bh) in enumerate(black_holes):
                dist = distance(pos_bh, pos_p)
                if dist < BH_PARTICLE_SWALLOW_RADIUS:
                    # BH swallows the particle => BH mass += m_p, remove particle
                    black_holes[b_index][0] += m_p  # increase BH's mass
                    # remove particle p_index
                    particles.pop(p_index)
                    any_swallowed = True
                    found_swallow = True
                    break
            if found_swallow:
                break

# =========================================================
# Offscreen Removal
# =========================================================
def remove_offscreen():
    """
    Remove any BH or particle that goes outside [-SIZE, SIZE].
    """
    # Filter BH
    bh_survivors = []
    for (m, pos, vel) in black_holes:
        (x,y) = pos
        if -SIZE <= x <= SIZE and -SIZE <= y <= SIZE:
            bh_survivors.append([m, pos, vel])
    black_holes[:] = bh_survivors

    # Filter particles
    p_survivors = []
    for (m, pos, vel) in particles:
        (x,y) = pos
        if -SIZE <= x <= SIZE and -SIZE <= y <= SIZE:
            p_survivors.append([m, pos, vel])
    particles[:] = p_survivors

# =========================================================
# Object Creation & Removal
# =========================================================
def add_black_hole(mass, x, y, vx=0.0, vy=0.0):
    black_holes.append([mass, [x,y], [vx,vy]])

def add_black_hole_in_orbit(x, y):
    """
    Place a new BH with mass=DEFAULT_BH_MASS in orbit around nearest existing BH.
    If no BH exist, just place it at rest.
    """
    if not black_holes:
        add_black_hole(DEFAULT_BH_MASS, x, y, 0.0, 0.0)
        return
    # Find nearest BH
    nearest_i = None
    min_d = float("inf")
    for i,(m_bh,pos_bh,vel_bh) in enumerate(black_holes):
        d = distance(pos_bh,[x,y])
        if d < min_d:
            min_d = d
            nearest_i = i
    m_bh, pos_bh, vel_bh = black_holes[nearest_i]
    dx = x - pos_bh[0]
    dy = y - pos_bh[1]
    r = np.hypot(dx,dy) or 1e-9
    # approximate circular orbit velocity ignoring the new BH's mass
    v_mag = np.sqrt(G*m_bh/r)
    vx_new = -v_mag*(dy/r)
    vy_new =  v_mag*(dx/r)
    add_black_hole(DEFAULT_BH_MASS, x,y, vx_new, vy_new)

def add_particle(mass, x, y, vx, vy):
    particles.append([mass, [x,y], [vx,vy]])

def add_particle_stationary(x, y):
    add_particle(DEFAULT_PARTICLE_MASS, x, y, 0.0,0.0)

def add_particle_in_orbit(x,y):
    """
    Particle orbits nearest BH
    """
    if not black_holes:
        add_particle_stationary(x,y)
        return
    nearest_i = None
    min_d = float("inf")
    for i,(m_bh,pos_bh,vel_bh) in enumerate(black_holes):
        d = distance(pos_bh,[x,y])
        if d<min_d:
            min_d = d
            nearest_i = i
    m_bh, pos_bh, vel_bh = black_holes[nearest_i]
    dx = x - pos_bh[0]
    dy = y - pos_bh[1]
    r = np.hypot(dx,dy) or 1e-9
    v_mag = np.sqrt(G*m_bh/r)
    vx = -v_mag*(dy/r)
    vy =  v_mag*(dx/r)
    add_particle(DEFAULT_PARTICLE_MASS, x,y, vx, vy)

def add_particle_custom(x,y, v_mag, angle_deg):
    theta = np.radians(angle_deg)
    vx = v_mag*np.cos(theta)
    vy = v_mag*np.sin(theta)
    add_particle(DEFAULT_PARTICLE_MASS, x,y, vx,vy)

def remove_nearest_object(x, y, threshold=REMOVE_THRESHOLD):
    """
    Remove the nearest BH or particle if it's within 'threshold' distance.
    """
    nearest_d = float("inf")
    nearest_is_bh = None
    nearest_i = None

    # BH check
    for i,(m,pos,vel) in enumerate(black_holes):
        d = distance(pos,[x,y])
        if d < nearest_d:
            nearest_d = d
            nearest_is_bh = True
            nearest_i = i

    # Particle check
    for i,(m,pos,vel) in enumerate(particles):
        d = distance(pos,[x,y])
        if d < nearest_d:
            nearest_d = d
            nearest_is_bh = False
            nearest_i = i

    if nearest_d<threshold and nearest_i is not None:
        if nearest_is_bh:
            black_holes.pop(nearest_i)
        else:
            particles.pop(nearest_i)

# =========================================================
# Main GUI Class
# =========================================================
class GravitySimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Merging BH & Particles with Mass Gains")

        # 1) Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.set_facecolor("black")
        self.ax.set_aspect("equal","box")
        self.ax.set_xlim(-SIZE,SIZE)
        self.ax.set_ylim(-SIZE,SIZE)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # background "stars"
        num_stars=150
        star_x = np.random.uniform(-1.5*SIZE,1.5*SIZE,num_stars)
        star_y = np.random.uniform(-1.5*SIZE,1.5*SIZE,num_stars)
        self.ax.scatter(star_x,star_y,c='white',s=2,alpha=0.6,zorder=0)

        # BH & Particle scatter
        self.bh_scatter = self.ax.scatter([],[], s=120, c='red', edgecolor='black', zorder=5)
        self.p_scatter = self.ax.scatter([],[], s=15, c='cyan', zorder=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=12)

        # Mouse event
        self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

        # 2) Control Panel
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.grid(row=0, column=1, sticky="nw", padx=8, pady=8)

        tk.Label(ctrl_frame, text="Click Mode:", font=("Arial",10,"bold")
                ).grid(row=0, column=0, columnspan=2)

        # 6 modes
        self.mode_var = tk.StringVar(value="bh_stationary")
        tk.Radiobutton(ctrl_frame, text="Add BH (Stationary)",
                       variable=self.mode_var, value="bh_stationary"
                      ).grid(row=1, column=0, columnspan=2, sticky="w")

        tk.Radiobutton(ctrl_frame, text="Add BH (Orbit)",
                       variable=self.mode_var, value="bh_orbit"
                      ).grid(row=2, column=0, columnspan=2, sticky="w")

        tk.Radiobutton(ctrl_frame, text="Add Particle (Stationary)",
                       variable=self.mode_var, value="p_stationary"
                      ).grid(row=3, column=0, columnspan=2, sticky="w")

        tk.Radiobutton(ctrl_frame, text="Add Particle (Orbit)",
                       variable=self.mode_var, value="p_orbit"
                      ).grid(row=4, column=0, columnspan=2, sticky="w")

        tk.Radiobutton(ctrl_frame, text="Add Particle (Custom)",
                       variable=self.mode_var, value="p_custom"
                      ).grid(row=5, column=0, columnspan=2, sticky="w")

        tk.Radiobutton(ctrl_frame, text="Remove Object",
                       variable=self.mode_var, value="remove"
                      ).grid(row=6, column=0, columnspan=2, sticky="w")

        # Particle velocity input
        tk.Label(ctrl_frame,text="p_v_mag:").grid(row=7,column=0,sticky="e")
        self.p_vmag_var = tk.DoubleVar(value=2.0)
        tk.Entry(ctrl_frame, textvariable=self.p_vmag_var, width=8
                ).grid(row=7,column=1,pady=2)

        tk.Label(ctrl_frame,text="p_angle°:").grid(row=8,column=0,sticky="e")
        self.p_angle_var = tk.DoubleVar(value=0.0)
        tk.Entry(ctrl_frame, textvariable=self.p_angle_var, width=8
                ).grid(row=8,column=1,pady=2)

        # 3) Window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 4) Begin simulation
        self.update_sim_id = None
        self.update_sim()

    def on_plot_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x_click, y_click = event.xdata, event.ydata
        mode = self.mode_var.get()

        if mode == "bh_stationary":
            # Add BH with default mass, no velocity
            add_black_hole(DEFAULT_BH_MASS, x_click, y_click)

        elif mode == "bh_orbit":
            add_black_hole_in_orbit(x_click, y_click)

        elif mode == "p_stationary":
            add_particle_stationary(x_click, y_click)

        elif mode == "p_orbit":
            add_particle_in_orbit(x_click,y_click)

        elif mode == "p_custom":
            v_mag = self.p_vmag_var.get()
            angle_deg = self.p_angle_var.get()
            add_particle_custom(x_click, y_click, v_mag, angle_deg)

        elif mode == "remove":
            remove_nearest_object(x_click, y_click, REMOVE_THRESHOLD)

        self.refresh_scatters()

    def update_sim(self):
        """
        Steps:
         1) Compute acceleration & Euler-update for BHs
         2) Compute acceleration & Euler-update for Particles
         3) BH-BH merges
         4) Particle-particle merges
         5) BH swallows particles
         6) (Optional) remove offscreen objects
         7) Refresh & schedule next update
        """
        # 1) BH update
        bh_accels = []
        for i in range(len(black_holes)):
            a = compute_acceleration_on_object(i, is_bh=True)
            bh_accels.append(a)
        for i in range(len(black_holes)):
            m,pos,vel = black_holes[i]
            vel_new = vel + bh_accels[i]*DT
            pos_new = np.array(pos) + vel_new*DT
            black_holes[i] = [m, pos_new.tolist(), vel_new.tolist()]

        # 2) Particle update
        p_accels = []
        for i in range(len(particles)):
            a = compute_acceleration_on_object(i, is_bh=False)
            p_accels.append(a)
        for i in range(len(particles)):
            m,pos,vel = particles[i]
            vel_new = vel + p_accels[i]*DT
            pos_new = np.array(pos) + vel_new*DT
            particles[i] = [m, pos_new.tolist(), vel_new.tolist()]

        # 3) BH merges
        merge_bh_bh()
        # 4) Particle merges
        merge_particle_particle()
        # 5) BH swallows particles
        bh_swallow_particles()
        # 6) Remove offscreen (optional)
        remove_offscreen()

        # 7) redraw
        self.refresh_scatters()
        self.update_sim_id = self.root.after(20, self.update_sim)

    def refresh_scatters(self):
        # black holes
        if black_holes:
            bh_xy = np.array([bh[1] for bh in black_holes])
            self.bh_scatter.set_offsets(bh_xy)
        else:
            self.bh_scatter.set_offsets(np.empty((0,2)))

        # particles
        if particles:
            p_xy = np.array([p[1] for p in particles])
            self.p_scatter.set_offsets(p_xy)
        else:
            self.p_scatter.set_offsets(np.empty((0,2)))

        self.canvas.draw_idle()

    def on_closing(self):
        if self.update_sim_id is not None:
            self.root.after_cancel(self.update_sim_id)
        self.root.destroy()

# =========================================================
# Main
# =========================================================
if __name__=="__main__":
    root = tk.Tk()
    app = GravitySimulatorGUI(root)
    root.mainloop()
