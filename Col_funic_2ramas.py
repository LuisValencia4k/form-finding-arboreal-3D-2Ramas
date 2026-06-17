"""
===============================================================================
MOTOR DE FORM-FINDING FEM 3D: TEOREMA DE DESCOMPOSICIÓN ORTOGONAL
===============================================================================
Autor: Luis Alberto Valencia Pérez (Independent Researcher)
Fecha: Junio 2026
DOI de la Publicación: https://doi.org/10.5281/zenodo.20706928
Licencia: MIT License

Descripción:
Algoritmo de relajación angular dinámica para nodos arbóreos asimétricos de
hormigón armado. Implementa el cálculo del Jacobiano analítico para separar la
componente geométrica del Residuo de Compatibilidad Elástica (T_hiper) y 
evalúa la seguridad del nodo mediante MCFT (Modified Compression Field Theory).
===============================================================================
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

class FormFindingFEM:
    def __init__(self):
        self.nodes = []      
        self.elements = []   
        self.K_global = np.zeros((0, 0))
        self.F_global = np.zeros(0)
        self.u_global = np.zeros(0)
        self.n_dof = 0
        self.converged = False
        self.history_Tres = []
        self.initial_node_coords = None   

        self.default_ks = 5.0/6.0   

    # --------------------------------------------------------------
    # 1. MALLADO Y CONDICIONES DE BORDE
    # --------------------------------------------------------------
    def add_node(self, x, y, z, fix=None, load=None):
        if fix is None:
            fix = [0,0,0,0,0,0]
        if load is None:
            load = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.nodes.append({
            'x': float(x), 'y': float(y), 'z': float(z),
            'fix': fix,
            'load': np.array(load, dtype=float)
        })
        return len(self.nodes)-1

    def add_element(self, i, j, b, h, E, G, is_branch=False, is_trunk=False, is_rigid_link=False,
                    top_node=None, L_xy=None, alpha_inicial_deg=None):
        xi, yi, zi = self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']
        xj, yj, zj = self.nodes[j]['x'], self.nodes[j]['y'], self.nodes[j]['z']
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        L = math.sqrt(dx*dx + dy*dy + dz*dz)

        if is_branch:
            if top_node is None:
                raise ValueError("En ramas hijas debe indicarse 'top_node'.")
            node_sup = i if i == top_node else j
            node_inf = j if i == top_node else i
            
            xs, ys = self.nodes[node_sup]['x'], self.nodes[node_sup]['y']
            xi_inf, yi_inf = self.nodes[node_inf]['x'], self.nodes[node_inf]['y']
            if L_xy is None:
                L_xy = math.sqrt((xs - xi_inf)**2 + (ys - yi_inf)**2)
            if alpha_inicial_deg is None:
                dz_inicial = self.nodes[node_sup]['z'] - self.nodes[node_inf]['z']
                alpha_inicial_rad = math.atan2(dz_inicial, L_xy)
                alpha_inicial_deg = math.degrees(alpha_inicial_rad)
            else:
                alpha_inicial_rad = math.radians(alpha_inicial_deg)
                
            branch_params = {
                'node_sup': node_sup,
                'node_inf': node_inf,
                'L_xy': L_xy,
                'alpha_rad': alpha_inicial_rad,
                'alpha_deg': alpha_inicial_deg
            }
        else:
            branch_params = None

        A = b * h
        Iy, Iz = (h * b**3) / 12.0, (b * h**3) / 12.0
        a_max, b_min = max(b, h), min(b, h)
        J = (1/3.0) * a_max * b_min**3 * (1 - 0.63 * (b_min / a_max))
        Asy = Asz = A * self.default_ks

        element = {
            'i': i, 'j': j,
            'dx': dx, 'dy': dy, 'dz': dz, 'L': L,
            'b': b, 'h': h, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J,
            'Asy': Asy, 'Asz': Asz,
            'E': E, 'G': G,
            'is_branch': is_branch,
            'is_trunk': is_trunk,
            'is_rigid_link': is_rigid_link,
            'branch_params': branch_params
        }
        self.elements.append(element)
        return len(self.elements)-1

    # --------------------------------------------------------------
    # 2. MATRICES DE RIGIDEZ LOCAL Y TRANSFORMACIÓN
    # --------------------------------------------------------------
    def _k_local_3d_timoshenko(self, E, G, A, Iy, Iz, J, Asy, Asz, L):
        k = np.zeros((12,12))
        EA_L, GJ_L = E * A / L, G * J / L
        phi_y = 12 * E * Iz / (G * Asy * L**2)
        phi_z = 12 * E * Iy / (G * Asz * L**2)

        a_z = (12 * E * Iz) / (L**3 * (1 + phi_y))
        b_z = (6 * E * Iz)  / (L**2 * (1 + phi_y))
        c_z = (4 + phi_y) * E * Iz / (L * (1 + phi_y))
        d_z = (2 - phi_y) * E * Iz / (L * (1 + phi_y))
        
        a_y = (12 * E * Iy) / (L**3 * (1 + phi_z))
        b_y = (6 * E * Iy)  / (L**2 * (1 + phi_z))
        c_y = (4 + phi_z) * E * Iy / (L * (1 + phi_z))
        d_y = (2 - phi_z) * E * Iy / (L * (1 + phi_z))

        # 1. Axial y Torsión
        k[0,0] = k[6,6] = EA_L
        k[0,6] = k[6,0] = -EA_L
        k[3,3] = k[9,9] = GJ_L
        k[3,9] = k[9,3] = -GJ_L

        # 2. Flexión Plano XY (v, theta_z)
        k[1,1] = k[7,7] = a_z
        k[1,7] = k[7,1] = -a_z
        k[1,5] = k[5,1] = b_z
        k[1,11] = k[11,1] = b_z
        k[5,7] = k[7,5] = -b_z    # CORREGIDO: acoplamiento cruzado negativo
        k[7,11] = k[11,7] = -b_z  # CORREGIDO: acoplamiento cruzado negativo
        k[5,5] = k[11,11] = c_z
        k[5,11] = k[11,5] = d_z

        # 3. Flexión Plano XZ (w, theta_y)
        # Nota: La regla de la mano derecha invierte el patrón de signos aquí.
        k[2,2] = k[8,8] = a_y
        k[2,8] = k[8,2] = -a_y
        k[2,4] = k[4,2] = -b_y    # CORREGIDO
        k[2,10] = k[10,2] = -b_y  # CORREGIDO
        k[4,8] = k[8,4] = b_y     # CORREGIDO
        k[8,10] = k[10,8] = b_y   # CORREGIDO
        k[4,4] = k[10,10] = c_y
        k[4,10] = k[10,4] = d_y

        return k

    def _matriz_transformacion_3d(self, dx, dy, dz, L):
        cx, cy, cz = dx/L, dy/L, dz/L
        vec_x = np.array([cx, cy, cz])
        if abs(cz) > 0.9999:
            vec_y = np.array([0,1,0])
            vec_z = np.array([-1,0,0]) if cz > 0 else np.array([1,0,0])
        else:
            vec_Z_glob = np.array([0,0,1])
            vec_y = np.cross(vec_Z_glob, vec_x)
            vec_y = vec_y / np.linalg.norm(vec_y)
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)
        lam = np.vstack([vec_x, vec_y, vec_z])
        T = np.zeros((12,12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = lam
        return T

    # --------------------------------------------------------------
    # 3. ENSAMBLAJE Y SOLUCIÓN DEL SISTEMA GLOBAL
    # --------------------------------------------------------------
    def _update_element_geometry(self, e):
        i, j = e['i'], e['j']
        e['dx'] = self.nodes[j]['x'] - self.nodes[i]['x']
        e['dy'] = self.nodes[j]['y'] - self.nodes[i]['y']
        e['dz'] = self.nodes[j]['z'] - self.nodes[i]['z']
        e['L'] = math.sqrt(e['dx']**2 + e['dy']**2 + e['dz']**2)

    def ensamblar_sistema_global(self):
        self.n_dof = 6 * len(self.nodes)
        self.K_global = np.zeros((self.n_dof, self.n_dof))
        self.F_global = np.zeros(self.n_dof)

        for idx, node in enumerate(self.nodes):
            self.F_global[6*idx:6*idx+6] = node['load']

        for e in self.elements:
            self._update_element_geometry(e)
            T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
            k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'], e['Asy'], e['Asz'], e['L'])
            K_elem = T.T @ k_loc @ T

            dofs = [6*e['i'] + k for k in range(6)] + [6*e['j'] + k for k in range(6)]
            for r in range(12):
                for c in range(12):
                    self.K_global[dofs[r], dofs[c]] += K_elem[r, c]

        for idx, node in enumerate(self.nodes):
            dof0 = 6*idx
            for d in range(6):
                if node['fix'][d] == 1:
                    self.K_global[dof0+d, :] = 0
                    self.K_global[:, dof0+d] = 0
                    self.K_global[dof0+d, dof0+d] = 1.0
                    self.F_global[dof0+d] = 0.0

    def resolver(self):
        try:
            self.u_global = np.linalg.solve(self.K_global, self.F_global)
        except np.linalg.LinAlgError:
            self.u_global = np.linalg.lstsq(self.K_global, self.F_global, rcond=None)[0]

    # --------------------------------------------------------------
    # 4. CÁLCULO DEL RESIDUO ORTOGONAL GLOBAL T_res
    # --------------------------------------------------------------
    def obtener_fuerzas_nodales_elemento(self, e):
        self._update_element_geometry(e)
        u_elem = np.concatenate([self.u_global[6*e['i']:6*e['i']+6], self.u_global[6*e['j']:6*e['j']+6]])
        T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
        k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'], e['Asy'], e['Asz'], e['L'])
        return T.T @ (k_loc @ (T @ u_elem))

    def compute_T_res_global(self):
        T_vec = self.compute_T_res_vec()
        return np.linalg.norm(T_vec)

    def compute_T_res_vec(self):
        """Devuelve el vector residuo (3 componentes) en kN."""
        T_vec = np.zeros(3)
        for e in self.elements:
            if not e.get('is_branch', False):
                continue
            f_glob = self.obtener_fuerzas_nodales_elemento(e)
            bp = e['branch_params']
            
            # Fuerzas en el nodo inferior
            if bp['node_inf'] == e['i']:
                F = f_glob[0:3]
                u_hat = np.array([e['dx'], e['dy'], e['dz']]) / e['L']
            else:
                F = f_glob[6:9]
                u_hat = np.array([-e['dx'], -e['dy'], -e['dz']]) / e['L']

            F_paralelo = np.dot(F, u_hat) * u_hat
            F_perp = F - F_paralelo
            T_vec += F_perp
        return T_vec

    # --------------------------------------------------------------
    # 5. BUCLE PRINCIPAL DE FORM-FINDING
    # --------------------------------------------------------------
    def form_finding_loop(self, max_iter=10000, tol=1e-3, w_max=0.1, w_min=0.01, lam=0.1, verbose=True):
        # Guardar geometría inicial (para visualización posterior)
        self.initial_node_coords = deepcopy([{'x':n['x'], 'y':n['y'], 'z':n['z']} for n in self.nodes])
        
        # Historiales
        self.history_Tres = []      # norma del residuo
        self.history_Gvec = []      # vector residuo (3 componentes)
        self.history_Vcol = []      # cortante en el fuste (columna padre)
        self.history_alphas = []

        # --- RASTREO DEL MÍNIMO HISTÓRICO ---
        best_Tres = float('inf')
        best_iter = 0
        best_node_coords = None     # snapshot de coordenadas en el mínimo
        best_branch_params = None   # snapshot de ángulos en el mínimo

        for it in range(max_iter):
            # 1. Ensamblar y resolver el sistema FEM
            self.ensamblar_sistema_global()
            self.resolver()

            # --- NUEVO: Guardar ángulos actuales antes de modificarlos ---
            current_alphas = [e['branch_params']['alpha_rad'] for e in self.elements if e.get('is_branch', False)]
            if current_alphas:
                self.history_alphas.append(current_alphas)
            
            # 2. Calcular vector residuo completo G(alpha) = (Gx, Gy, Gz)
            G_vec = self.compute_T_res_vec()
            T_res = np.linalg.norm(G_vec)
            
            # 3. Calcular cortante en el fuste (columna padre)
            V_col = 0.0
            for e in self.elements:
                if e.get('is_trunk', False):
                    f_glob = self.obtener_fuerzas_nodales_elemento(e)
                    F_tope = f_glob[6:9]    # Fuerzas en el nodo superior (tope del fuste)
                    V_col = math.sqrt(F_tope[0]**2 + F_tope[1]**2)
                    break
            
            # 4. Guardar en historiales
            self.history_Tres.append(T_res)
            self.history_Gvec.append(G_vec.copy())
            self.history_Vcol.append(V_col)

            # --- SNAPSHOT DEL MÍNIMO HISTÓRICO ---
            # Se guarda ANTES de actualizar ángulos, cuando la geometría actual
            # produce el T_res más bajo visto hasta ahora.
            if T_res < best_Tres:
                best_Tres = T_res
                best_iter = it
                best_node_coords = deepcopy([{'x':n['x'], 'y':n['y'], 'z':n['z']} for n in self.nodes])
                best_branch_params = deepcopy([
                    e['branch_params'] for e in self.elements if e.get('is_branch', False)
                ])

            if verbose:
                print(f"\nIter {it:2d}: T_res = {T_res:.4e} kN, V_fuste = {V_col:.4f} kN")
                print(f"           G = ({G_vec[0]:8.4f}, {G_vec[1]:8.4f}, {G_vec[2]:8.4f}) kN")
            
            # 5. Recorrer elementos para actualizar ángulos (ramas) y mostrar información
            for e in self.elements:
                # -------- RAMAS HIJAS (optimización) --------
                if e.get('is_branch', False):
                    bp = e['branch_params']
                    f_glob = self.obtener_fuerzas_nodales_elemento(e)
                    
                    # Fuerza en el nodo inferior (arranque)
                    F = f_glob[0:3] if bp['node_inf'] == e['i'] else f_glob[6:9]
                    F_horiz = math.sqrt(F[0]**2 + F[1]**2)
                    
                    # Ángulo funicular objetivo (desde la horizontal)
                    theta_F = math.atan2(abs(F[2]), F_horiz)
                    alpha_old = bp['alpha_rad']
                    
                    # Cálculo de la componente ortogonal local (solo para imprimir)
                    u_hat = np.array([e['dx'], e['dy'], e['dz']]) / e['L']
                    if bp['node_inf'] != e['i']:
                        u_hat = -u_hat
                    F_paralelo = np.dot(F, u_hat) * u_hat
                    F_perp = F - F_paralelo
                    F_perp_mag = np.linalg.norm(F_perp)
                    
                    if verbose:
                        print(f"  Rama {bp['node_sup']} -> theta_F: {math.degrees(theta_F):.4f}° | alpha_old: {math.degrees(alpha_old):.4f}° | F_perp local: {F_perp_mag:.4f} kN")
                    
                    # Relajación dinámica
                    w_i = w_max * math.exp(-lam * it) + w_min
                    alpha_new = alpha_old + w_i * (theta_F - alpha_old)
                    bp['alpha_rad'] = alpha_new
                    bp['alpha_deg'] = math.degrees(alpha_new)
                    
                    # Actualizar coordenada Z del nodo inferior
                    Z_sup = self.nodes[bp['node_sup']]['z']
                    self.nodes[bp['node_inf']]['z'] = Z_sup - bp['L_xy'] * np.tan(alpha_new)
                
                # EVALUAR COLUMNA PADRE (Monitoreo de hiperestaticidad)
                elif e.get('is_trunk', False):
                    f_glob = self.obtener_fuerzas_nodales_elemento(e)
                    F_tope = f_glob[6:9]  # Fuerzas en el nodo tope del fuste
                    V_col = math.sqrt(F_tope[0]**2 + F_tope[1]**2)
                    if verbose:
                        print(f"  Columna Padre -> Cortante absorbido (Hiperestaticidad): {V_col:.4f} kN")

            # --- PARADA TEMPRANA ---
            # Validar divergencia solo después de una fase inicial de estabilización
            if it > 500:
                # Si el residuo sube un 15% por encima del mínimo histórico, cortamos
                if T_res > best_Tres * 1.15:
                    if verbose:
                        print(f"\n[Early Stopping] Divergencia detectada. Deteniendo iteraciones en la {it}.")
                    break
            # --------------------------------------------------

            if T_res < tol:
                if verbose:
                    print(f"\nConvergencia lograda en iteración {it}! (T_res = {T_res:.3e} kN)")
                self.converged = True
                break

        # --- RESTAURAR EL ESTADO DEL MÍNIMO HISTÓRICO ---
        # Si el algoritmo se pasó del mínimo (oscilación), regresamos al mejor estado.
        last_Tres = self.history_Tres[-1] if self.history_Tres else float('inf')
        if best_node_coords is not None and best_Tres < last_Tres:
            # Restaurar coordenadas nodales
            for idx, snap in enumerate(best_node_coords):
                self.nodes[idx]['x'] = snap['x']
                self.nodes[idx]['y'] = snap['y']
                self.nodes[idx]['z'] = snap['z']
            # Restaurar parámetros de rama si están disponibles
            if best_branch_params is not None:
                branch_idx = 0
                for e in self.elements:
                    if e.get('is_branch', False):
                        if branch_idx < len(best_branch_params):
                            e['branch_params'] = deepcopy(best_branch_params[branch_idx])
                        branch_idx += 1
                        
            # Re-resolver para dejar K_global y u_global consistentes con el estado óptimo
            self.ensamblar_sistema_global()
            self.resolver()
            
            # --- NUEVO: SINCRONIZAR HISTORIALES ---
            # Añadimos el estado óptimo como punto final para que las gráficas 
            # caigan al mínimo exacto y los post-procesos lean el valor correcto.
            self.history_Tres.append(best_Tres)
            self.history_Vcol.append(self.history_Vcol[best_iter] if self.history_Vcol else 0.0)
            self.history_alphas.append([bp['alpha_rad'] for bp in best_branch_params] if best_branch_params else [])
            # -----------------------------------------

            if verbose:
                improvement = (last_Tres - best_Tres) / last_Tres * 100
                print(f"\n[Rollback] Restaurado al mínimo histórico:")
                print(f"  Iteración del mínimo : {best_iter}")
                print(f"  T_res en el mínimo   : {best_Tres:.4f} kN")
                print(f"  T_res al terminar    : {last_Tres:.4f} kN")
                print(f"  Mejora por rollback  : {improvement:.2f}%")
        elif verbose:
            print(f"\n[Sin rollback] El último estado ya es el mínimo (iter {best_iter}).")

        return self.history_Tres
    
    # --------------------------------------------------------------
    # 7. POST-PROCESO: VERIFICACIÓN MCFT
    # --------------------------------------------------------------
    def post_process_MCFT(self, T_res_kN, f_c_MPa, E_s_MPa, A_s_mm2, A_biela_mm2, V_u_kN, theta_deg):
        """
        Evalúa la capacidad del nodo tridimensional utilizando la Teoría del Campo 
        de Compresión Modificada (MCFT), considerando el residuo T_res como 
        fuerza de hendimiento tridimensional.
        """
        # Conversión de T_res a Newtons para consistencia de unidades con MPa y mm2
        T_res_N = T_res_kN * 1000.0
        
        # Cálculo de deformación unitaria transversal (ε_t)
        eps_t = T_res_N / (E_s_MPa * A_s_mm2)
        
        # Factor de eficiencia del puntal (β_s) con límite máximo de 1.0
        beta_s = (E_s_MPa * A_s_mm2) / (0.8 * E_s_MPa * A_s_mm2 + 170.0 * T_res_N)
        beta_s = min(beta_s, 1.0)
        
        # Capacidad nominal del puntal de concreto a compresión (F_nn) en kN
        F_nn_N = 0.85 * f_c_MPa * A_biela_mm2 * beta_s
        F_nn_kN = F_nn_N / 1000.0
        
        # Proyección vertical para resistir el cortante último
        theta_rad = math.radians(theta_deg)
        Capacidad_vertical_kN = F_nn_kN * math.sin(theta_rad)
        
        # Dictamen de seguridad
        margen = Capacidad_vertical_kN / V_u_kN if V_u_kN > 0 else 0
        status = "CUMPLE ✓" if Capacidad_vertical_kN >= V_u_kN else "NO CUMPLE ✗"
        
        print("\n=== POST-PROCESO: VERIFICACIÓN MCFT ===")
        print(f"Demanda neta de hendimiento (|T_res|) : {T_res_kN:.2f} kN")
        print(f"Deformación transversal (ε_t)         : {eps_t:.6f} mm/mm")
        print(f"Factor de eficiencia (β_s)            : {beta_s:.3f}")
        print(f"Capacidad nominal del puntal (F_nn)   : {F_nn_kN:.2f} kN")
        print(f"Capacidad vertical disponible         : {Capacidad_vertical_kN:.2f} kN")
        print(f"Cortante último actuante (V_u)        : {V_u_kN:.2f} kN")
        print(f"Ratio Demanda/Capacidad               : {1/margen:.2f}" if margen > 0 else "N/A")
        print(f"Estado de Verificación                : {status}")
        print("=====================================================================")
        
        return beta_s, Capacidad_vertical_kN
    
    # --------------------------------------------------------------
    # 8. DIAGNÓSTICO ENERGÉTICO (ÍNDICE FUNICULAR)
    # --------------------------------------------------------------
    def diagnostico_funicular(self):
        U_axial = 0.0
        U_flex = 0.0

        for e in self.elements:
            # Ignoramos los enlaces rígidos virtuales del capitel
            if e.get('is_rigid_link', False):
                continue

            self._update_element_geometry(e)
            i, j = e['i'], e['j']
            
            # Extraer desplazamientos globales del elemento
            u_elem_global = np.concatenate([self.u_global[6*i:6*i+6], self.u_global[6*j:6*j+6]])
            
            # Matrices de transformación y rigidez local
            T = self._matriz_transformacion_3d(e['dx'], e['dy'], e['dz'], e['L'])
            k_loc = self._k_local_3d_timoshenko(e['E'], e['G'], e['A'], e['Iy'], e['Iz'], e['J'], e['Asy'], e['Asz'], e['L'])
            
            # Pasar a coordenadas locales
            u_loc = T @ u_elem_global
            f_loc = k_loc @ u_loc  

            # Energía de Deformación: U = 0.5 * f^T * u
            # Energía Axial (grados de libertad 0 y 6: traslación en el eje X local)
            u_ax = 0.5 * (f_loc[0]*u_loc[0] + f_loc[6]*u_loc[6])
            
            # Energía Flexional/Cortante/Torsión (el resto de los grados de libertad)
            u_fl = 0.5 * np.sum(f_loc[1:6] * u_loc[1:6]) + 0.5 * np.sum(f_loc[7:12] * u_loc[7:12])
            
            U_axial += u_ax
            U_flex += u_fl

        U_total = U_axial + U_flex
        if U_total == 0:
            return

        pct_axial = (U_axial / U_total) * 100
        pct_flex = (U_flex / U_total) * 100
        indice_funicular = U_axial / U_total
        
        # Clasificación tipológica
        if indice_funicular >= 0.95:
            clasif = "ESTRUCTURA FUNICULAR"
        elif indice_funicular >= 0.85:
            clasif = "CUASI FUNICULAR"
        elif indice_funicular >= 0.60:
            clasif = "SISTEMA HÍBRIDO (FLEXO-COMPRESIÓN)"
        else:
            clasif = "ESTRUCTURA A FLEXIÓN DOMINANTE"

        # --- CORRECCIÓN DE FORMATO DE ENERGÍA ---
        # Escalado dinámico para evitar el truncamiento a "0.00 J"
        if U_total < 1e-3:
            energia_str = f"{U_total * 1e6:.2f} µJ"
        elif U_total < 1.0:
            energia_str = f"{U_total * 1000:.2f} mJ"
        else:
            energia_str = f"{U_total:.2f} J"

        print("\n================================================")
        print("            DIAGNÓSTICO FUNICULAR               ")
        print("================================================")
        print(f"T_res global remanente = {self.history_Tres[-1]:.2f} kN")
        print(f"Energía de deformación = {energia_str} (Bruto: {U_total:.4e} J)")
        print("------------------------------------------------")
        print(f"Energía axial          = {pct_axial:.1f} %")
        print(f"Energía flexión/corte  = {pct_flex:.1f} %")
        print(f"\nÍndice funicular       = {indice_funicular:.3f}")
        print(f"\nClasificación:")
        print(f"[{pct_axial:.1f}%] {clasif}")
        print("================================================")

    # --------------------------------------------------------------
    # 9. OPTIMIZADOR MORFOLÓGICO (SECCIONES CIRCULARES AHUSADAS)
    # --------------------------------------------------------------
    def optimizar_dimensiones_circulares(self, f_c_MPa=30.0, h_losa_m=0.25, E_concreto_Pa=30e9):
        """
        Optimizador de dimensionamiento basado en el Método Valencia-Paleólogo.
        Arranca con un diámetro de 10 cm y lo incrementa iterativamente hasta 
        satisfacer: Punzonamiento (losa), Pandeo (ramas) y Flexocompresión (fuste).
        """
        print("\n================================================")
        print("  OPTIMIZACIÓN MORFOLÓGICA (CÍRCULOS AHUSADOS)  ")
        print("================================================")
        
        d_losa_efectivo = h_losa_m - 0.04 # Peralte efectivo de la losa
        phi_punz = 0.75
        phi_comp = 0.65
        
        # Resistencia a cortante del concreto (MPa a N/m2)
        v_c_Pa = 0.33 * math.sqrt(f_c_MPa) * 1e6 
        f_c_Pa = f_c_MPa * 1e6

        # Variables de diseño (Inician en 10 cm)
        D_padre = 0.10
        d_hija_tope = 0.10
        D_hija_base = 0.10
        step = 0.05 # Incrementos de 5 cm

        # ---------------------------------------------------------
        # FASE 1: Punzonamiento en Losa -> Define 'd' (cúspide)
        # ---------------------------------------------------------
        # Extraemos la carga vertical máxima (V_u) que llega a los nodos superiores
        Vu_max_punz = 0.0
        for node in self.nodes:
            if node['load'][2] < 0: # Cargas aplicadas hacia abajo
                Vu_max_punz = max(Vu_max_punz, abs(node['load'][2]))
        Vu_max_N = Vu_max_punz * 1000.0

        while True:
            # Perímetro crítico circular: b_o = pi * (d + d_losa/2 + d_losa/2)
            b_o = math.pi * (d_hija_tope + d_losa_efectivo)
            V_resistido = phi_punz * v_c_Pa * b_o * d_losa_efectivo
            if V_resistido >= Vu_max_N:
                break
            d_hija_tope += step

        # ---------------------------------------------------------
        # FASE 2: Pandeo en las Ramas -> Define el ahusamiento (D_hija_base)
        # ---------------------------------------------------------
        D_hija_base = d_hija_tope # Arranca igual al tope
        Pu_rama_max = 0.0
        L_rama_max = 0.0

        # Encontrar la demanda máxima de compresión axial en las ramas
        for e in self.elements:
            if e.get('is_branch', False):
                f_glob = self.obtener_fuerzas_nodales_elemento(e)
                F_axial = np.linalg.norm(f_glob[0:3]) # Magnitud de la fuerza axial
                Pu_rama_max = max(Pu_rama_max, F_axial)
                L_rama_max = max(L_rama_max, e['L'])
        
        Pu_rama_N = Pu_rama_max * 1000.0

        while True:
            # Inercia promedio para sección ahusada (usando el diámetro a la mitad)
            d_mid = (D_hija_base + d_hija_tope) / 2.0
            I_avg = (math.pi * d_mid**4) / 64.0
            
            # Carga crítica de pandeo de Euler (K=1.0 para bi-articulada segura)
            P_cr = (math.pi**2 * E_concreto_Pa * I_avg) / (L_rama_max**2)
            
            if 0.75 * P_cr >= Pu_rama_N: # Factor de seguridad contra pandeo
                break
            D_hija_base += step

        # ---------------------------------------------------------
        # FASE 3: Flexocompresión -> Define 'D' (Fuste Padre)
        # ---------------------------------------------------------
        D_padre = max(D_padre, D_hija_base) # El padre no puede ser más delgado que el inicio de la rama
        
        # Extraer demandas sobre la columna padre
        Pu_padre_N = 0.0
        Mu_padre_Nm = 0.0
        
        for e in self.elements:
            if e.get('is_trunk', False):
                f_glob = self.obtener_fuerzas_nodales_elemento(e)
                Pu_padre_N = abs(f_glob[2]) * 1000.0 # Axial Z
                
                # Cortante hiperestático que genera el momento en la base
                V_hiper_N = math.sqrt(f_glob[6]**2 + f_glob[7]**2) * 1000.0
                Mu_padre_Nm = V_hiper_N * e['L'] # M = V * H
                break

        while True:
            Area = (math.pi * D_padre**2) / 4.0
            S_modulo = (math.pi * D_padre**3) / 32.0
            
            # Esfuerzo máximo de compresión por flexión y carga axial combinadas
            esfuerzo_max = (Pu_padre_N / Area) + (Mu_padre_Nm / S_modulo)
            
            # Capacidad aproximada de compresión (0.85 f'c) afectada por phi
            esfuerzo_resistente = phi_comp * 0.85 * f_c_Pa
            
            # Condición constructiva: El padre debe ser capaz de alojar las dos ramas
            condicion_geometrica = D_padre >= (1.5 * D_hija_base) 
            
            if esfuerzo_max <= esfuerzo_resistente and condicion_geometrica:
                break
            D_padre += step

        # ---------------------------------------------------------
        # REPORTE DE DISEÑO
        # ---------------------------------------------------------
        print("         RESULTADOS DEL DIMENSIONAMIENTO        ")
        print("------------------------------------------------")
        print(f"Fuste Padre (D)        : {D_padre*100:.1f} cm   [Gobierna: Flexocompresión & Geometría]")
        print(f"Rama Hija Base (D_inf) : {D_hija_base*100:.1f} cm   [Gobierna: Pandeo de Euler]")
        print(f"Rama Hija Tope (d_sup) : {d_hija_tope*100:.1f} cm   [Gobierna: Punzonamiento Losa]")
        print("------------------------------------------------")
        print(f"Ratio de ahusamiento de rama : {d_hija_tope/D_hija_base:.2f}")
        print("================================================\n")
        
        return D_padre, D_hija_base, d_hija_tope

    # --------------------------------------------------------------
    # 6. VISUALIZACIÓN 3D Y GRÁFICO DE CONVERGENCIA
    # --------------------------------------------------------------
    def plot_results(self, show_initial=True):
        plt.figure(figsize=(12,5))
        
        # Figura 1: convergencia
        plt.subplot(1,2,1)
        plt.semilogy(self.history_Tres, 'o-', color='#2b6cb0', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('T_res (kN)')
        plt.title('Minimización del Residuo Nodal (Hendimiento)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # Figura 2: estructura 3D
        ax = plt.subplot(1,2,2, projection='3d')
        for e in self.elements:
            i, j = e['i'], e['j']
            xi, yi, zi = self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']
            xj, yj, zj = self.nodes[j]['x'], self.nodes[j]['y'], self.nodes[j]['z']
            
            if e.get('is_branch', False):
                color = 'blue'
                lbl = 'Rama hija'
            elif e['E'] > 1e11: # Detección de enlace rígido
                color = 'black'
                lbl = 'Capitel (Rígido)'
            else:
                color = 'red'
                lbl = 'Columna padre'
                
            ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, linewidth=2, label=lbl)

        for node in self.nodes:
            ax.scatter(node['x'], node['y'], zs=node['z'], c='black', s=30, depthshade=True)

        if show_initial and self.initial_node_coords:
            for e in self.elements:
                if e.get('is_branch', False):
                    i, j = e['i'], e['j']
                    xi0, yi0, zi0 = self.initial_node_coords[i]['x'], self.initial_node_coords[i]['y'], self.initial_node_coords[i]['z']
                    xj0, yj0, zj0 = self.initial_node_coords[j]['x'], self.initial_node_coords[j]['y'], self.initial_node_coords[j]['z']
                    ax.plot([xi0, xj0], [yi0, yj0], [zi0, zj0], 'gray', linestyle=':', linewidth=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)') #type: ignore
        ax.set_title('Morfología Arbórea Final')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_field_2d(self, resolution=25, margin_deg=10.0):
        """
        Genera un mapa de curvas de nivel de T_res y un campo vectorial (quiver)
        del gradiente descendente respecto a los ángulos de 2 ramas.
        """
        ramas = [e for e in self.elements if e.get('is_branch', False)]
        if len(ramas) != 2:
            print("\n[Aviso] El gráfico de gradiente 2D está diseñado para exactamente 2 ramas.")
            return

        print("\nGenerando mapa topográfico y campo vectorial del gradiente 2D...")
        print("Resolviendo el sistema FEM sobre la malla espacial. Esto puede tomar unos segundos...")

        # Obtener el estado final (estacionario) de los ángulos
        a1_final = ramas[0]['branch_params']['alpha_rad']
        a2_final = ramas[1]['branch_params']['alpha_rad']

        # Definir los límites de la malla en radianes
        margin_rad = math.radians(margin_deg)
        a1_vals = np.linspace(a1_final - margin_rad, a1_final + margin_rad, resolution)
        a2_vals = np.linspace(a2_final - margin_rad, a2_final + margin_rad, resolution)
        A1, A2 = np.meshgrid(a1_vals, a2_vals)
        Z_Tres = np.zeros_like(A1)

        # Guardar el estado original profundo para no corromper el modelo
        orig_nodes = deepcopy(self.nodes)
        orig_a1, orig_a2 = a1_final, a2_final

        # Evaluar T_res en toda la malla
        for i in range(resolution):
            for j in range(resolution):
                # Asignar ángulos de la malla
                ramas[0]['branch_params']['alpha_rad'] = A1[i, j]
                ramas[1]['branch_params']['alpha_rad'] = A2[i, j]

                # Actualizar geometría Z de los nodos inferiores
                for rama in ramas:
                    bp = rama['branch_params']
                    Z_sup = self.nodes[bp['node_sup']]['z']
                    self.nodes[bp['node_inf']]['z'] = Z_sup - bp['L_xy'] * math.tan(bp['alpha_rad'])

                # Resolver FEM y obtener magnitud del residuo
                self.ensamblar_sistema_global()
                self.resolver()
                Z_Tres[i, j] = self.compute_T_res_global()

        # Restaurar el estado original del solver
        self.nodes = deepcopy(orig_nodes)
        ramas[0]['branch_params']['alpha_rad'] = orig_a1
        ramas[1]['branch_params']['alpha_rad'] = orig_a2
        self.ensamblar_sistema_global()
        self.resolver()

        # Calcular el gradiente espacial (dZ/dA2, dZ/dA1)
        # Invertimos el signo para que los vectores apunten en dirección de descenso (hacia el mínimo)
        dZ_dA2, dZ_dA1 = np.gradient(Z_Tres, a2_vals, a1_vals)
        U, V = -dZ_dA1, -dZ_dA2 

        # Normalizar los vectores para una mejor visualización en el gráfico quiver
        M = np.hypot(U, V)
        M[M == 0] = 1.0  # Evitar división por cero
        Un, Vn = U/M, V/M

        # --- CREACIÓN DEL GRÁFICO ---
        plt.figure(figsize=(10, 8))
        
        # 1. Curvas de nivel rellenas (Contourf)
        cp = plt.contourf(np.degrees(A1), np.degrees(A2), Z_Tres, levels=40, cmap='viridis', alpha=0.85)
        plt.colorbar(cp, label='Magnitud del Residuo Nodal $T_{res}$ (kN)')
        
        # 2. Líneas de contorno
        plt.contour(np.degrees(A1), np.degrees(A2), Z_Tres, levels=20, colors='black', linewidths=0.5, alpha=0.6)
        
        # 3. Campo vectorial (Quiver)
        plt.quiver(np.degrees(A1), np.degrees(A2), Un, Vn, M, cmap='autumn', pivot='mid', scale=35, width=0.003, alpha=0.9)

        # 4. Trayectoria de iteraciones
        if hasattr(self, 'history_alphas') and len(self.history_alphas) > 0:
            # Filtrar historial para mostrar solo la parte que cae dentro del gráfico
            hist_a1 = [math.degrees(a[0]) for a in self.history_alphas]
            hist_a2 = [math.degrees(a[1]) for a in self.history_alphas]
            plt.plot(hist_a1, hist_a2, color='white', linestyle='--', linewidth=2, label='Trayectoria Dinámica')
            plt.plot(hist_a1[0], hist_a2[0], 'wo', markeredgecolor='black', markersize=8, label='Arranque Inicial')

        # 5. Punto Estacionario (Final del script)
        plt.plot(math.degrees(a1_final), math.degrees(a2_final), 'r*', markersize=18, markeredgecolor='black', label='Estado Final (Punto Estacionario)')

        # 6. Mínimo absoluto encontrado en la malla
        min_idx = np.unravel_index(np.argmin(Z_Tres, axis=None), Z_Tres.shape)
        a1_min_grid = math.degrees(a1_vals[min_idx[1]])
        a2_min_grid = math.degrees(a2_vals[min_idx[0]])
        plt.plot(a1_min_grid, a2_min_grid, 'kX', markersize=12, label='Mínimo Teórico Local')

        # Configuración de ejes
        plt.xlabel(r'Ángulo $\alpha_1$ de la Rama 1 (grados)', fontsize=11)
        plt.ylabel(r'Ángulo $\alpha_2$ de la Rama 2 (grados)', fontsize=11)
        plt.title('Superficie de Energía $T_{res}$ y Campo Vectorial del Gradiente', fontsize=13, pad=15)
        
        # Ajustar límites de visualización al margen seleccionado
        plt.xlim([math.degrees(a1_vals[0]), math.degrees(a1_vals[-1])])
        plt.ylim([math.degrees(a2_vals[0]), math.degrees(a2_vals[-1])])
        
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()


# =============================================================================
# FUNCIONES AUXILIARES PARA EXPERIMENTOS (sin modificar la clase)
# =============================================================================

def build_arboreal_structure(base_fix, capitel_rigido=True, enlace_extra=False):
    """
    Construye la estructura arbórea con condiciones de borde variables.
    - base_fix: lista de 6 ints (0/1) para los DOF de la base.
    - capitel_rigido: si True, usa elementos rígidos (E=30e12) entre nodo1 y arranques;
                      si False, usa rigidez normal (30e9).
    - enlace_extra: si True, añade un elemento rígido entre los nodos de arranque (2 y 3).
    Retorna un objeto FormFindingFEM ya configurado (sin ejecutar optimización).
    """
    solver = FormFindingFEM()
    
    # Nodos de la columna
    solver.add_node(0.0, 0.0, 0.0, fix=base_fix)  # Nodo 0: base con condiciones variables
    solver.add_node(0.0, 0.0, 1.5)                # Nodo 1: tope columna central
    
    # SOLUCIÓN: Separación de 20 cm en el eje Y para evitar longitud cero en el enlace extra
    solver.add_node(0.0, 0.1, 2.5)                # Nodo 2: arranque rama1
    solver.add_node(0.0, -0.1, 2.5)               # Nodo 3: arranque rama2
    
    # Nodos superiores (llegada a la losa)
    solver.add_node(2.0, 0.0, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-600.0,0,0,0]) # Nodo 4
    solver.add_node(-1.5, 1.2, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-600.0,0,0,0])# Nodo 5
    
    # Elementos
    solver.add_element(0, 1, b=0.4, h=0.4, E=30e9, G=12e9, is_trunk=True)
    
    # Enlaces capitel
    E_cap = 30e12 if capitel_rigido else 30e9
    G_cap = 12e12 if capitel_rigido else 12e9   # Bug corregido: G debe escalar con E
    solver.add_element(1, 2, b=1.0, h=1.0, E=E_cap, G=G_cap, is_rigid_link=True)
    solver.add_element(1, 3, b=1.0, h=1.0, E=E_cap, G=G_cap, is_rigid_link=True)
    
    # Enlace extra entre arranques (simula unión rígida adicional horizontal)
    if enlace_extra:
        solver.add_element(2, 3, b=1.0, h=1.0, E=30e12, G=12e12)
        
    # Ramas (sin optimización, ángulos fijos en 35°)
    solver.add_element(2, 4, b=0.3, h=0.3, E=30e9, G=12e9, is_branch=True, top_node=4, alpha_inicial_deg=35.0)
    solver.add_element(3, 5, b=0.3, h=0.3, E=30e9, G=12e9, is_branch=True, top_node=5, alpha_inicial_deg=35.0)
    
    return solver

def get_hiper_reactions(solver):
    """Devuelve (V_fuste, M_fuste) en kN y kN·m a partir del análisis estático.
    
    Usa las fuerzas en el nodo TOPE (nodo j, índices 6:9 y 9:12) para ser
    consistente con el monitoreo del form_finding_loop.
    """
    solver.ensamblar_sistema_global()
    solver.resolver()
    for e in solver.elements:
        if e.get('is_trunk', False):
            f_glob = solver.obtener_fuerzas_nodales_elemento(e)
            # Nodo tope = nodo j → índices 6:9 (fuerzas) y 9:12 (momentos)
            F_tope = f_glob[6:9]
            M_tope = f_glob[9:12]
            V = np.sqrt(F_tope[0]**2 + F_tope[1]**2) / 1000.0   # kN
            M = np.sqrt(M_tope[0]**2 + M_tope[1]**2) / 1000.0   # kN·m
            return V, M
    return 0.0, 0.0

def compute_jacobian_alpha(solver, delta=1e-4):
    """
    Calcula la matriz jacobiana J_G (3 x n_ramas) en la configuración actual.
    delta en radianes.
    Retorna (J, G0_vec, grad, norm_grad) donde grad = J^T * G0_vec.
    """
    # Guardar estado original
    original_nodes = deepcopy(solver.nodes)
    original_alphas = []
    branch_elems = []
    for e in solver.elements:
        if e.get('is_branch', False):
            branch_elems.append(e)
            original_alphas.append(e['branch_params']['alpha_rad'])
    n = len(branch_elems)
    if n == 0:
        return None, None, None, None
    # Resolver sistema actual (para G0)
    solver.ensamblar_sistema_global()
    solver.resolver()
    G0_vec = solver.compute_T_res_vec()  # vector (3,)
    J = np.zeros((3, n))
    
    for i, e in enumerate(branch_elems):
        # Perturbar alpha_i
        alpha_new = original_alphas[i] + delta
        e['branch_params']['alpha_rad'] = alpha_new
        # Actualizar geometría: recalcular Z del nodo inferior
        bp = e['branch_params']
        Z_sup = solver.nodes[bp['node_sup']]['z']
        Z_inf_new = Z_sup - bp['L_xy'] * np.tan(alpha_new)
        solver.nodes[bp['node_inf']]['z'] = Z_inf_new
        # Re-ensamblar y resolver
        solver.ensamblar_sistema_global()
        solver.resolver()
        G_pert = solver.compute_T_res_vec()
        # Restaurar estado original para la siguiente perturbación
        solver.nodes = deepcopy(original_nodes)
        for e2, a_orig in zip(branch_elems, original_alphas):
            e2['branch_params']['alpha_rad'] = a_orig
        solver.ensamblar_sistema_global()
        solver.resolver()
        # Columna del jacobiano
        J[:, i] = (G_pert - G0_vec) / delta
    
    # Calcular gradiente y su norma DESPUÉS de tener J completo
    grad = J.T @ G0_vec
    norm_grad = np.linalg.norm(grad)
    
    return J, G0_vec, grad, norm_grad



def run_experiments():
    """
    Ejecuta los experimentos propuestos:
    Experimento A: correlación con hiperestaticidad (cambios en apoyos y uniones).
                   Compara V_fuste ANTES y DESPUÉS de optimizar para ver si Thiper
                   es realmente de origen hiperestático o algo más sutil.
    Experimento B: liberación progresiva de grados de libertad, mismo esquema A/D.
    Estudio del jacobiano: sobre la geometría optimizada del caso original.
    Prueba de descenso: un paso de gradiente para verificar la dirección de mejora.
    """
    from typing import TypedDict
    class OptParams(TypedDict):
        max_iter: int
        tol: float
        w_max: float
        w_min: float
        lam: float
        verbose: bool

    OPT_PARAMS: OptParams = {
        "max_iter": 10000,
        "tol": 1e-2,
        "w_max": 0.1,
        "w_min": 0.01,
        "lam": 0.1,
        "verbose": False,
    }

    print("\n" + "="*70)
    print("EXPERIMENTO A: CORRELACIÓN CON EL GRADO DE HIPERESTATICIDAD")
    print("  [V_inicial = ángulos a 35°]  [V_final = post-optimización]")
    print("="*70)
    # Casos en orden creciente de hiperestaticidad
    casos = [
        ("Caso 1: Base articulada, sin capitel rígido", [1,1,1,0,0,0], False, False),
        ("Caso 2: Base articulada, con capitel rígido", [1,1,1,0,0,0], True,  False),
        ("Caso 3: Base empotrada, capitel rígido",      [1,1,1,1,1,1], True,  False),
        ("Caso 4: Base empotrada + enlace extra",       [1,1,1,1,1,1], True,  True),
    ]
    resultados_A = []
    for desc, fix, cap_rig, enlace in casos:
        # Estado inicial (α = 35°, sin optimizar)
        s0 = build_arboreal_structure(fix, capitel_rigido=cap_rig, enlace_extra=enlace)
        V0, M0 = get_hiper_reactions(s0)
        T0 = s0.compute_T_res_global()

        # Estado post-optimización
        s1 = build_arboreal_structure(fix, capitel_rigido=cap_rig, enlace_extra=enlace)
        s1.form_finding_loop(**OPT_PARAMS)
        Vf, Mf = get_hiper_reactions(s1)
        Tf = s1.history_Tres[-1] if s1.history_Tres else float('nan')

        resultados_A.append((desc, V0, M0, T0, Vf, Mf, Tf))
        print(f"\n{desc}")
        print(f"  Antes  : V={V0:.2f} kN  M={M0:.2f} kN·m  T_res={T0:.2f} kN")
        print(f"  Después: V={Vf:.2f} kN  M={Mf:.2f} kN·m  T_res={Tf:.2f} kN")
        red_V = (V0 - Vf) / V0 * 100 if V0 > 0 else 0.0
        red_T = (T0 - Tf) / T0 * 100 if T0 > 0 else 0.0
        print(f"  Reducción: ΔV={red_V:.1f}%  ΔT_res={red_T:.1f}%")

    print("\n" + "="*70)
    print("EXPERIMENTO B: LIBERACIÓN PROGRESIVA DE GRADOS DE LIBERTAD")
    print("  [V_inicial = ángulos a 35°]  [V_final = post-optimización]")
    print("="*70)
    configs = [
        ("Empotrado (todos fijos)",               [1,1,1,1,1,1]),
        ("Liberar rotación X",                    [1,1,1,0,1,1]),
        ("Liberar rotaciones X y Y",              [1,1,1,0,0,1]),
        ("Liberar rotaciones X,Y,Z (articulado)", [1,1,1,0,0,0]),
        ("Liberar también traslación X",          [0,1,1,0,0,0]),
        ("Liberar también traslación Y",          [0,0,1,0,0,0]),
    ]
    resultados_B = []
    for desc, fix in configs:
        s0 = build_arboreal_structure(fix, capitel_rigido=True, enlace_extra=False)
        V0, M0 = get_hiper_reactions(s0)
        T0 = s0.compute_T_res_global()

        s1 = build_arboreal_structure(fix, capitel_rigido=True, enlace_extra=False)
        s1.form_finding_loop(**OPT_PARAMS)
        Vf, Mf = get_hiper_reactions(s1)
        Tf = s1.history_Tres[-1] if s1.history_Tres else float('nan')

        resultados_B.append((desc, V0, M0, T0, Vf, Mf, Tf))
        print(f"\n{desc}")
        print(f"  Antes  : V={V0:.2f} kN  M={M0:.2f} kN·m  T_res={T0:.2f} kN")
        print(f"  Después: V={Vf:.2f} kN  M={Mf:.2f} kN·m  T_res={Tf:.2f} kN")
        red_V = (V0 - Vf) / V0 * 100 if V0 > 0 else 0.0
        red_T = (T0 - Tf) / T0 * 100 if T0 > 0 else 0.0
        print(f"  Reducción: ΔV={red_V:.1f}%  ΔT_res={red_T:.1f}%")

    # ============================================================
    # ESTUDIO DEL JACOBIANO SOBRE GEOMETRÍA OPTIMIZADA
    # ============================================================
    print("\n" + "="*70)
    print("ESTUDIO DEL JACOBIANO (SOBRE LA GEOMETRÍA OPTIMIZADA)")
    print("="*70)

    solver_opt = FormFindingFEM()
    solver_opt.add_node(0.0, 0.0, 0.0, fix=[1,1,1,1,1,1])
    solver_opt.add_node(0.0, 0.0, 1.5)
    solver_opt.add_node(0.0, 0.0, 2.5)
    solver_opt.add_node(0.0, 0.0, 2.5)
    solver_opt.add_node(2.0, 0.0, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-600.0,0,0,0])
    solver_opt.add_node(-1.5, 1.2, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-600.0,0,0,0])
    solver_opt.add_element(0, 1, b=0.4, h=0.4, E=30e9,  G=12e9,  is_trunk=True)
    solver_opt.add_element(1, 2, b=1.0, h=1.0, E=30e12, G=12e12, is_rigid_link=True)
    solver_opt.add_element(1, 3, b=1.0, h=1.0, E=30e12, G=12e12, is_rigid_link=True)
    solver_opt.add_element(2, 4, b=0.3, h=0.3, E=30e9,  G=12e9,  is_branch=True, top_node=4, alpha_inicial_deg=35.0)
    solver_opt.add_element(3, 5, b=0.3, h=0.3, E=30e9,  G=12e9,  is_branch=True, top_node=5, alpha_inicial_deg=35.0)

    solver_opt.form_finding_loop(max_iter=10000, tol=1e-2, w_max=0.1, w_min=0.01, lam=0.1, verbose=False)
    
    # Calcular jacobiano, gradiente, etc.
    J, G0_vec, grad, norm_grad = compute_jacobian_alpha(solver_opt, delta=1e-4)
    if J is not None and G0_vec is not None:
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        rank = np.sum(S > 1e-6)
        nul_dim = 3 - rank
        print(f"  Gradiente J^T G = {grad}")
        print(f"  Norma del gradiente = {norm_grad:.6f} kN²/rad")
        print(f"Jacobiano J_G (3 x {J.shape[1]}):")
        print(f"  Rango = {rank}")
        print(f"  Dimensión del núcleo izquierdo (Nul(J_G^T)) = {nul_dim}")
        print(f"  Valores singulares: {S}")
        print(f"  Vector residuo G(alpha*) = {G0_vec} kN")
        print(f"  Norma total del residuo: {np.linalg.norm(G0_vec):.6f} kN")
        if rank < 3:
            null_left = U[:, rank:]
            proj_norm = np.linalg.norm(null_left.T @ G0_vec) if null_left.size > 0 else 0.0
            print(f"  Norma de la proyección del residuo sobre Nul(J_G^T): {proj_norm:.6f} kN")
        else:
            print("  El jacobiano tiene rango completo.")
    else:
        print("  No se encontraron ramas para calcular el jacobiano.")
        return
    
    # ============================================================
    # PRUEBA DE DESCENSO CON UN PASO DE GRADIENTE
    # ============================================================
    print("\n" + "="*70)
    print("PRUEBA DE DESCENSO CON UN PASO DE GRADIENTE (η variable)")
    print("="*70)
    
    # Guardar ángulos actuales y geometría original (solo coordenadas)
    alphas_current = []
    branch_elems = []
    for e in solver_opt.elements:
        if e.get('is_branch', False):
            alphas_current.append(e['branch_params']['alpha_rad'])
            branch_elems.append(e)
    original_alphas = alphas_current.copy()
    # Guardar coordenadas originales de todos los nodos
    original_coords = [{'x': n['x'], 'y': n['y'], 'z': n['z']} for n in solver_opt.nodes]
    
    # Valores de η a probar (en rad²/kN)
    eta_list = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    best_eta = None
    best_T_new = float('inf')
    best_reduction = 0.0
    
    T_old = np.linalg.norm(G0_vec)
    Phi_old = 0.5 * T_old**2
    
    print(f"Estado inicial (después de optimización): T_res = {T_old:.6f} kN")
    
    for eta in eta_list:
        # Dirección de descenso: delta_alpha = -eta * grad
        grad_arr = np.asarray(grad)
        delta_alpha = -eta * grad_arr   # grad es J^T G (vector 2x1)
        alphas_new = [alphas_current[i] + delta_alpha[i] for i in range(len(alphas_current))]
        
        # Actualizar geometría en solver_opt (ángulos y coordenadas Z)
        for i, e in enumerate(branch_elems):
            bp = e['branch_params']
            bp['alpha_rad'] = alphas_new[i]
            bp['alpha_deg'] = math.degrees(alphas_new[i])
            Z_sup = solver_opt.nodes[bp['node_sup']]['z']
            Z_inf_new = Z_sup - bp['L_xy'] * np.tan(alphas_new[i])
            solver_opt.nodes[bp['node_inf']]['z'] = Z_inf_new
        
        # Resolver con la nueva geometría
        solver_opt.ensamblar_sistema_global()
        solver_opt.resolver()
        G_new = solver_opt.compute_T_res_vec()
        T_new = np.linalg.norm(G_new)
        reduc = (T_old - T_new) / T_old * 100
        
        print(f"\nη = {eta:.2e}: T_res = {T_new:.6f} kN, reducción = {reduc:.4f}%")
        
        if T_new < best_T_new:
            best_T_new = T_new
            best_eta = eta
            best_reduction = reduc
        
        # Restaurar geometría original (coordenadas y ángulos)
        for idx, node in enumerate(solver_opt.nodes):
            node['x'] = original_coords[idx]['x']
            node['y'] = original_coords[idx]['y']
            node['z'] = original_coords[idx]['z']
        for i, e in enumerate(branch_elems):
            e['branch_params']['alpha_rad'] = original_alphas[i]
            e['branch_params']['alpha_deg'] = math.degrees(original_alphas[i])
        # Resolver una vez para dejar el solver en el estado original (opcional)
        solver_opt.ensamblar_sistema_global()
        solver_opt.resolver()
    
    print("\n" + "-"*70)
    if best_eta is not None and best_reduction > 0:
        print(f"Mejor η = {best_eta:.2e} con reducción de T_res del {best_reduction:.4f}%")
        print("✓ El gradiente apunta en dirección de descenso (funcional disminuye).")
    else:
        print("El funcional no disminuyó con ningún η. Revisar cálculo de gradiente o signo.")
    print("="*70 + "\n")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Opción: ejecutar experimentos en lugar del caso original
    RUN_EXPERIMENTS = False # Cambiar a False para ejecutar la optimización y gráficos 
    
    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        # Código de optimización y gráficos
        solver = FormFindingFEM()

        # NODOS
        solver.add_node(0.0, 0.0, 0.0, fix=[1,1,1,1,1,1]) # 0. Base empotrada
        solver.add_node(0.0, 0.0, 1.5)                    # 1. Tope de la columna padre

        # Nodos inferiores independientes (Arranque escalonado)
        solver.add_node(0.0, 0.0, 2.5)                    # 2. Arranque Rama 1
        solver.add_node(0.0, 0.0, 2.5)                    # 3. Arranque Rama 2

        # Nodos en losa superior (Carga puntual hacia abajo, z liberado para permitir acortamiento)
        solver.add_node(2.0, 0.0, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-300.0,0,0,0])   # 4. Rama 1 top
        solver.add_node(-1.5, 1.2, 5.0, fix=[1,1,0,1,1,1], load=[0,0,-300.0,0,0,0])  # 5. Rama 2 top

        # ELEMENTOS
        solver.add_element(0, 1, b=0.4, h=0.4, E=30e9, G=12e9, is_trunk=True)  # Columna Padre

        # Enlaces rígidos (Simulan el bloque de concreto del capitel)
        solver.add_element(1, 2, b=1.0, h=1.0, E=30e12, G=12e12, is_rigid_link=True)
        solver.add_element(1, 3, b=1.0, h=1.0, E=30e12, G=12e12, is_rigid_link=True)

        # Ramas sujetas a optimización
        solver.add_element(2, 4, b=0.3, h=0.3, E=30e9, G=12e9, is_branch=True, top_node=4, alpha_inicial_deg=35.0)
        solver.add_element(3, 5, b=0.3, h=0.3, E=30e9, G=12e9, is_branch=True, top_node=5, alpha_inicial_deg=35.0)

        # EJECUCIÓN DEL ALGORITMO
        # max_iter=10000: límite empírico del experimento de convergencia (ajustar según estudio)
        history = solver.form_finding_loop(max_iter=10000, tol=1e-2, w_max=0.1, w_min=0.01, lam=0.1)

        # REPORTE
        print("\n--- RESULTADOS FINALES ---")
        alpha_critico = 90.0
        for e in solver.elements:
            if e.get('is_branch', False):
                bp = e['branch_params']
                print(f"Rama conectada al Nodo Sup {bp['node_sup']}: "
                      f"α = {bp['alpha_deg']:.2f}°, Cota de Arranque (Z_inf) = {solver.nodes[bp['node_inf']]['z']:.4f} m")
                alpha_critico = min(alpha_critico, bp['alpha_deg'])
        T_res_final = history[-1]

        # POST-PROCESO MCFT
        solver.post_process_MCFT(
            T_res_kN=T_res_final,
            f_c_MPa=30.0,
            E_s_MPa=200000.0,
            A_s_mm2=200.0,
            A_biela_mm2=200000.0,
            V_u_kN=400.0,
            theta_deg=alpha_critico
        )
        solver.diagnostico_funicular()

        # OPTIMIZACIÓN MORFOLÓGICA
        solver.optimizar_dimensiones_circulares(f_c_MPa=30.0, h_losa_m=0.25)
        solver.plot_results()
        solver.plot_gradient_field_2d(resolution=25, margin_deg=10.0)