"""
author: BEDEL Matéo (code original: Anthony Dard)
description: Reprise du code de Anthony Dard, et ajout des spines d'Hermite Cubique, 
des polynomes de Lagrange et ajout d'autres fonctionnalités dans le gui.
"""
from tkinter import *
import numpy as np
import math
import matplotlib.pyplot as plt


class CurveEditorWindow(Tk):

    def __init__(self, compute_algorithms) -> None:
        
        super().__init__()

        self.title("Curvy editor")
        self.geometry("1000x520")

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 5
        self.calc = None
        self.curve = None
        self.compute_algorithms = compute_algorithms

        self.rowconfigure([0, 1], weight=1, minsize=200)
        self.columnconfigure([0, 1], weight=1, minsize=200)

        #Paramètre des graphes
        self.params = {}
        self.show_graph = {"vectors":BooleanVar(value=False),"circle":BooleanVar(value=False)}

        self.setup_canvas()
        self.setup_panel()

    def setup_canvas(self):
        self.graph = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb")
        self.graph.grid(column=0,
                        padx=2,
                        pady=2,
                        rowspan=2,
                        columnspan=2,
                        sticky="nsew")

        self.graph.bind('<Button-1>', self.handle_canvas_click)
        self.graph.tag_bind("control_points", "<ButtonRelease-1>",
                            self.handle_drag_stop)
        self.graph.bind("<B1-Motion>", self.handle_drag)

    def setup_panel(self):
        # Right panel for options
        self.frame_pannel = Frame(self, relief=RAISED, bg="#e1e1e1")
        self.frame_curve_type = Frame(self.frame_pannel)
        self.frame_edit_type = Frame(self.frame_pannel)
        self.frame_edit_position = Frame(self.frame_pannel)
        self.frame_sliders = Frame(self.frame_pannel)

        # Selection of curve type
        curve_types = [algo['name'] for algo in self.compute_algorithms]
        curve_types_val = list(range(len(self.compute_algorithms)))
        self.curve_type = IntVar()
        self.curve_type.set(curve_types_val[0])

        self.radio_curve_buttons = [None] * len(self.compute_algorithms)
        for i in range(len(self.compute_algorithms)):
            self.radio_curve_buttons[i] = Radiobutton(self.frame_curve_type,
                                                      variable=self.curve_type,
                                                      text=curve_types[i],
                                                      value=curve_types_val[i],
                                                      bg="#e1e1e1")
            self.radio_curve_buttons[i].pack(side='left', expand=1)
            self.radio_curve_buttons[i].bind(
                "<ButtonRelease-1>",
                lambda event: self.graph.after(100, lambda: self.switch_curve()))

        # Selection of edit mode
        edit_types = ['Add', 'Remove', 'Drag', 'Select']
        edit_types_val = ["add", "remove", "drag", "select"]
        self.edit_types = StringVar()
        self.edit_types.set(edit_types_val[0])

        self.radio_edit_buttons = [None] * 4
        for i in range(4):
            self.radio_edit_buttons[i] = Radiobutton(self.frame_edit_type,
                                                     variable=self.edit_types,
                                                     text=edit_types[i],
                                                     value=edit_types_val[i],
                                                     bg="#e1e1e1")
            self.radio_edit_buttons[i].pack(side='left', expand=1)
            self.radio_edit_buttons[i].bind(
                "<ButtonRelease-1>", lambda event: self.reset_selection())

        # Edit position of selected point widget
        self.label_pos_x = Label(self.frame_edit_position, text='x: ')
        self.label_pos_y = Label(self.frame_edit_position, text='y: ')
        self.pos_x = StringVar()
        self.pos_y = StringVar()
        self.entry_position_x = Entry(self.frame_edit_position,
                                      textvariable=self.pos_x)
        self.entry_position_y = Entry(self.frame_edit_position,
                                      textvariable=self.pos_y)
        self.label_pos_x.grid(row=0, column=0)
        self.entry_position_x.grid(row=0, column=1)
        self.label_pos_y.grid(row=1, column=0)
        self.entry_position_y.grid(row=1, column=1)

        self.entry_position_x.bind("<FocusOut>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-KP_Enter>", self.update_pos)

        self.entry_position_y.bind("<FocusOut>", self.update_pos)
        self.entry_position_y.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_y.bind("<KeyPress-KP_Enter>", self.update_pos)

        self.setup_params()


        self.frame_pannel.grid(row=0,
                               column=2,
                               padx=2,
                               pady=2,
                               rowspan=2,
                               sticky="nswe")
        self.frame_curve_type.pack()
        self.frame_edit_type.pack()
        self.frame_edit_position.pack()
        self.frame_sliders.pack()

        self.button_reset = Button(self.frame_pannel, text="Reset")
        self.button_reset.pack(side=BOTTOM, fill="x")
        self.button_reset.bind("<ButtonRelease-1>",
                               lambda event: self.graph.delete("all"))

    #Initialise les paramètres des courbes
    def setup_params(self):

        for p in self.params: self.params[p].destroy()

        # Slider for Resolution
        self.params["label_resolution"] = Label(self.frame_sliders, text="Resolution")
        self.params["slider_resolution"] = Scale(self.frame_sliders, from_=5, to=500, orient=HORIZONTAL, bg="#e1e1e1", command = lambda e: self.draw_graph())
        self.params["slider_resolution"].set(100)
        self.params["label_resolution"].grid(row=1, column=0,  sticky='sw')
        self.params["slider_resolution"].grid(row=1, column=1,  sticky='w')

        #Gui de Hermite
        if self.compute_algorithms[self.curve_type.get()]["name"] == "Hermite":

            #Tension
            self.params["label_tension"] = Label(self.frame_sliders, text="Tension c")
            self.params["slider_tension"] = Scale(self.frame_sliders, from_=0, to=1, digits=3, resolution =   0.01, orient=HORIZONTAL, bg="#e1e1e1", command= lambda e: self.draw_graph())
            self.params["slider_tension"].set(.5)
            self.params["label_tension"].grid(row=2, column=0, sticky='sw')
            self.params["slider_tension"].grid(row=2, column=1, sticky='w')

            #Spacer
            spacer = Label(self.frame_sliders, text="")  
            spacer.grid(row=3, column=0) 

            #Vecteurs
            self.params["check_vector"] = Checkbutton(self.frame_sliders, text="Vecteurs", variable=self.show_graph["vectors"], command = lambda : self.draw_graph())
            self.params["check_vector"].grid(row=4, column=0, sticky='w')

            #Cercle courbure
            self.params["check_circle"] = Checkbutton(self.frame_sliders, text="Courbure", variable=self.show_graph["circle"], command = lambda : self.draw_graph())
            self.params["check_circle"].grid(row=5, column=0, sticky='sw')
            self.params["slider_rayon"] = Scale(self.frame_sliders, from_=0, to=1, digits=3, resolution =  0.01, orient=HORIZONTAL, bg="#e1e1e1", command= lambda e: self.draw_graph())
            self.params["slider_rayon"].set(.5)
            self.params["slider_rayon"].grid(row=5, column=1, sticky='w')

            #Spacer
            spacer = Label(self.frame_sliders, text="")  
            spacer.grid(row=6, column=0) 

            #Graph de courbure
            self.params["bouton_courbure"] = Button(self.frame_sliders, text="MàJ graphe courbure", bg="#e1e1e1", command=lambda: self.draw_curvature_graph(np.linspace(0, 1, self.params["slider_resolution"].get())))
            self.params["bouton_courbure"].grid(row=7, column=0, columnspan=2)
        
    def switch_curve(self):
        self.setup_params()
        self.draw_graph()

    def get_points(self):
        points = []
        for item in self.graph.find_withtag("control_points"):
            coords = self.graph.coords(item)
            points.append([
                float(coords[0] + self.radius),
                float(coords[1] + self.radius)
            ])  # Ensure curve accuracy
        return points

    def create_point(self, x, y, color):
        """Create a token at the given coordinate in the given color"""
        item = self.graph.create_oval(x - self.radius,
                                      y - self.radius,
                                      x + self.radius,
                                      y + self.radius,
                                      outline=color,
                                      fill=color,
                                      tags="control_points")
        return item

    def draw_polygon(self):

        self.graph.delete("control_polygon")
        points = self.get_points()
        for i in range(0, len(points) - 1):
            self.graph.create_line(points[i][0],
                                   points[i][1],
                                   points[i + 1][0],
                                   points[i + 1][1],
                                   fill="black",
                                   tags="control_polygon")

    #Affiche le graphe de courbure de la courbe
    def draw_curvature_graph(self, T):
        if (self.calc is None): return
        plt.clf() 
        plt.plot([x for x in T], self.calc["courbure"])
        plt.title('Courbure du spline')
        plt.xlabel('u')
        plt.ylabel('Courbure')
        plt.show() #Affiche le graphe

    #Dessine les vecteurs de la courbe
    def draw_vectors(self, points):
        for i in range(len(points)):
                p = np.array(points)
                v = self.calc["vectors"]
                self.graph.create_line(p[i,0], p[i,1], v[i,0], v[i,1], fill="blue",  width=2, tags="curve", arrow=LAST)

    #Dessine le cercle de courbure
    def draw_curvature_circle(self, T):

        ind = min(int(self.params["slider_rayon"].get()*len(T)),len(T)-1) #indice de courbure

        radius = 1/self.calc["courbure"][ind]
        center = self.curve[ind] + self.calc["normal"]*radius
        self.graph.create_oval(center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius,outline="green",width=2,tags="curve")

    #Dessine la courbe
    def draw_curve(self, color="red"):
        for i in range(0, self.curve.shape[0] - 1):
            self.graph.create_line(self.curve[i, 0],self.curve[i, 1],self.curve[i + 1, 0],self.curve[i + 1, 1],fill=color,width=3,tags="curve")

    #Execute toute les algos et dessine toute les courbes
    def draw_all_graph(self, algos, T, points):

        colors = ["red","orange","green","blue"]

        for i,algo in enumerate(algos):
            self.calc = algo(np.array(points), T)

            if "curve" in self.calc:
                self.curve = self.calc["curve"]
                self.draw_curve(colors[i % len(colors)])
            
    #Dessine le graph
    def draw_graph(self):

        #Initialisation
        self.graph.delete("curve")
        points = self.get_points()
        if len(points) <= 1: return

        T = np.linspace(0, 1, self.params["slider_resolution"].get()) #Resolution des points
        algo = self.compute_algorithms[self.curve_type.get()]['algo'] #Algorithme(s) de calcul de courbe

        #Calcul de la courbe
        match self.compute_algorithms[self.curve_type.get()]["name"]:

            case "Superposition": #Dessiner toute les courbe superposer
                self.draw_all_graph(algo, T, points)
                return

            case "Hermite": 
                self.calc = algo(np.array(points), T, self.params["slider_tension"].get(), self.params["slider_rayon"].get())

            case _: 
                self.calc = algo(np.array(points), T)
        
        #Affichage de la courbe
        if "curve" in self.calc:
            self.curve = self.calc["curve"]
            self.draw_curve()

        #Affichage des vecteurs
        if "vectors" in self.calc and self.show_graph["vectors"].get(): 
            self.draw_vectors(points)

        #Affichage du cercle de courbure
        if "normal" and "courbure" in self.calc and self.show_graph["circle"].get():
            self.draw_curvature_circle(T)

    def find_closest_with_tag(self, x, y, radius, tag):
        distances = []
        for item in self.graph.find_withtag(tag):
            c = self.graph.coords(item)
            d = (x - c[0])**2 + (y - c[1])**2
            if d <= radius**2:
                distances.append((item, c, d))

        return min(distances,
                   default=(None, [0, 0], float("inf")),
                   key=lambda p: p[2])

    def reset_selection(self):
        if self._selected_data['item'] is not None:
            self.graph.itemconfig(self._selected_data['item'], fill='black')

        self._selected_data['item'] = None
        self._selected_data["x"] = 0
        self._selected_data["y"] = 0

    def handle_canvas_click(self, event):
        self.reset_selection()

        if self.edit_types.get() == "add":
            item = self.create_point(event.x, event.y, "black")
            self.update_pos_entry(item)

            points = self.get_points()

            if len(points) > 1:
                self.graph.create_line(points[-2][0],
                                       points[-2][1],
                                       points[-1][0],
                                       points[-1][1],
                                       fill="black",
                                       tag="control_polygon")
                self.draw_graph()

        elif self.edit_types.get() == "remove":
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.delete(self._selected_data['item'])

                self.draw_polygon()
                self.draw_graph()

        elif self.edit_types.get() == "drag":
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")

            if self._selected_data['item'] is not None:
                self._selected_data["x"] = event.x
                self._selected_data["y"] = event.y
                self.graph.move(self._selected_data['item'],
                                event.x - coords[0] - self.radius,
                                event.y - coords[1] - self.radius)

        else:
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.itemconfig(self._selected_data['item'],
                                      fill='orange')  # Mark as selected
                self.update_pos_entry(self._selected_data['item'])

    def handle_drag_stop(self, event):
        """End drag of an object"""
        if self.edit_types.get() != "drag":
            return
        self.reset_selection()

    def handle_drag(self, event):
        """Handle dragging of an object"""
        if self.edit_types.get() != "drag" or self._selected_data[
                'item'] is None or "control_points" not in self.graph.gettags(
                    self._selected_data['item']):
            return

        # compute how much the mouse has moved
        delta_x = event.x - self._selected_data["x"]
        delta_y = event.y - self._selected_data["y"]
        # move the object the appropriate amount
        self.graph.move(self._selected_data['item'], delta_x, delta_y)
        # record the new position
        self._selected_data["x"] = event.x
        self._selected_data["y"] = event.y

        self.update_pos_entry(self._selected_data['item'])
        self.draw_polygon()
        self.draw_graph()

    def update_pos_entry(self, item):
        coords = self.graph.coords(item)
        self.entry_position_x.delete(0, END)
        self.entry_position_x.insert(0, int(coords[0]))
        self.entry_position_y.delete(0, END)
        self.entry_position_y.insert(0, int(coords[1]))

    def update_pos(self, event):
        if self.edit_types.get(
        ) != "select" or self._selected_data['item'] is None:
            return

        coords = self.graph.coords(self._selected_data['item'])
        self.graph.move(self._selected_data['item'],
                        float(self.pos_x.get()) - coords[0],
                        float(self.pos_y.get()) - coords[1])

        self.draw_polygon()
        self.draw_graph()


#Algorithme DeCasteljau pour Bezier
def DeCasteljau(points, T):
    n = points.shape[0] - 1
    result = []

    for t in T:
        r = points.copy()
        for k in range(0, n):
            for i in range(0, n - k):
                r[i, :] = (1 - t) * r[i, :] + t * r[i + 1, :]

        result.append(r[0, :])
    

    return {"curve":np.array(result)}

#Calcul la dérivé première d'une courbe S
def calcDerivative(S, U):
    S_speed = [(S[k+1] - S[k-1])/(2*(U[k+1] - U[k])) for k in range(1, len(U)-1)]
    S_speed = [(S[1] - S[0]) / (U[1] - U[0])] + S_speed + [(S[-1] - S[-2]) / (U[-1] - U[-2])] #valeurs au bords
    return S_speed

#Calcul la dérivé seconde d'une courbe S
def calcSecondDerivative(S, U):
    S_acc = [(S[k+1] - 2*S[k] + S[k-1])/((U[k+1] - U[k])**2) for k in range(1, len(U)-1)]
    S_acc = [(S[2] - 2*S[1] + S[0]) / (U[2] - U[0])**2] + S_acc + [(S[-1] - 2*S[-2] + S[-3]) / (U[-1] - U[-3])**2] #valeurs au bords
    return S_acc

#Renvoi la courbure d'une courbe S
def calcCourbure(S, U):

    #Calculs des dérivées
    S_speed = calcDerivative(S,U)
    S_acc = calcSecondDerivative(S, U)
    S_speed_norm = [math.sqrt(S_speed[k][0]**2 + S_speed[k][1]**2) for k in range(len(S_speed))]

    courbure = [(S_speed[k][0]*S_acc[k][1] - S_speed[k][1]*S_acc[k][0])/(S_speed_norm[k]**3) for k in range(len(S_speed))]

    return courbure

#Renvoi le vecteur normal de la courbe S à l'indice t
def calcNormal(S, U, t):

    #Calculs de la dérivée première et sa norme
    S_speed = calcDerivative(S,U)
    S_speed_norm = [math.sqrt(S_speed[k][0]**2 + S_speed[k][1]**2) for k in range(len(S_speed))]

    return [-S_speed[t][1]/S_speed_norm[t], S_speed[t][0]/S_speed_norm[t]]

#Algorithme du Spline Hermite Cubique
def cubicHermiteSpline(points, T, c=0, rayon_perc=0):

    N = len(points) - 1

    #Paramétrisation (u_k)
    param = [ i for i in range(N+1)]

    #Valeur d'échantillonage global
    U = param[0] + T*(param[N]-param[0]) 

    #Tangentes
    m = [ (1-c)*(points[i+2]-points[i])/(param[i+2]-param[i]) for i in range(0,N-1)]
    m.insert(0, (1-c)*(points[1]-points[0])/(param[1]-param[0]))
    m.append((1-c)*(points[N]-points[N-1])/(param[N]-param[N-1]))

    #Polynomes d'hermite
    H = [
        lambda t: 2*(t**3) - 3*t**2 + 1,
        lambda t: -2*(t**3) + 3*t**2,           
        lambda t: (t**3) - 2*t**2 + t,
        lambda t: (t**3) - t**2
    ]

    S = []
    k = 0
    for u in U: #Iteration dans les valeur d'échantillonage global

        #On change l'intervalle de paramétrisation si le paramètre global est trop grand
        while u > param[k+1]: k+=1
        
        #Changement de variable du paramètre global en local
        t = (u - param[k])/(param[k+1]-param[k])

        #Calcul de la spline Hermite cubique en t
        P = points[k]*H[0](t) + points[k+1]*H[1](t) + (param[k+1]-param[k])*m[k]*H[2](t) + (param[k+1]-param[k])*m[k+1]*H[3](t)
        S.append(P)

    #Calcul de la courbure et du vecteur normal
    courbure = calcCourbure(S,U)
    normal = calcNormal(S,U, min(int(rayon_perc*len(U)),len(U)-1))

    return {"curve":np.array(S), "vectors": np.array(points + m), "courbure": np.array(courbure), "normal": np.array(normal)};

#Algorithme Aitken-Neville pour le polynome de Lagrange
def Lagrange(points, T):

    n = points.shape[0]
    x, y = points[:, 0], points[:, 1]
    result = []

    for t in T:  
        
        u = x[0] + t*(x[-1] - x[0])
        p = y.copy()

        for j in range(1, n):
            for i in range(0, n-j):
                if x[i+j] - x[i] == 0 : return {} #Regarder division par 0
                p[i] = ((u - x[i]) * p[i+1] - (u - x[i+j]) * p[i]) / (x[i+j] - x[i])

        result.append([u, p[0]])

    return {"curve": np.array(result)}


if __name__ == "__main__":
    window = CurveEditorWindow([
        {"name": "Bezier", "algo": DeCasteljau},
        {"name": "Hermite", "algo": cubicHermiteSpline},
        {"name": "Lagrange", "algo": Lagrange},
        {"name": "Superposition", "algo": [DeCasteljau, cubicHermiteSpline, Lagrange]}
    ])
    
    window.mainloop()
