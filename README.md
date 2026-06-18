[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706928.svg)](https://doi.org/10.5281/zenodo.20706928)
# Form-Finding Funicular de Estructuras Arbóreas 3D — 2 Ramas
Motor computacional para el form-finding funicular de estructuras arbóreas tridimensionales con dos ramas, desarrollado como implementación de referencia del artículo:

> Valencia Pérez, L. A. (2026). *Form-Finding Funicular de Estructuras Arbóreas Tridimensionales Mediante Relajación Angular Dinámica*. Zenodo. https://doi.org/10.5281/zenodo.20706928

---

## Descripción

El repositorio contiene un único script monolítico (`Col_funic_2ramas.py`) que implementa:

- **FEM de vigas Timoshenko** con 6 GDL por nodo, ensamble de rigidez global y condiciones de borde por empotramiento o articulación
- **Relajación Angular Dinámica (RAD)** para minimizar el residuo ortogonal $T_\text{res} = \|G(\boldsymbol{\alpha})\|$ ajustando los ángulos de elevación de las ramas
- **Jacobiano analítico** de $G$ respecto a los ángulos, utilizado para la descomposición ortogonal $G = T_\text{geo} + T_\text{hiper}$ mediante la Alternativa de Fredholm
- **Análisis espectral** del Jacobiano: valores singulares, número de condición $\kappa$ y rango estructural
- **Verificación MCFT** (Modified Compression Field Theory) de la fuerza de hendimiento en el nodo de bifurcación
- **Visualización 3D** de la morfología final y topografía del funcional $\Phi$ en el espacio de ángulos

El fundamento teórico completo — incluyendo la demostración de $G \in C^1(\mathcal{A})$, el Teorema de Condición Necesaria de Funicularidad y el Principio de Funicularidad Ortogonal — se desarrolla en el artículo complementario (en preparación).

---

## Requisitos

```
Python >= 3.8
numpy
matplotlib
```

## Uso

El script está diseñado para ejecutarse directamente. Los parámetros geométricos, materiales y de carga se configuran en el bloque `if __name__ == '__main__'` al final del archivo:

```bash
python Col_funic_2ramas.py
```

Los parámetros principales que el usuario puede modificar son:

| Parámetro | Descripción |
|---|---|
| `P1`, `P2` | Cargas verticales en los nodos superiores (kN) |
| `alpha1_0`, `alpha2_0` | Ángulos de elevación iniciales (°) |
| `L_xy_1`, `L_xy_2` | Proyecciones horizontales de las ramas (m) |
| `b`, `h` | Sección transversal de las ramas (m) |
| `E`, `G` | Módulo elástico y de cortante (kN/m²) |
| `omega_max`, `omega_min`, `lam` | Parámetros de la relajación angular dinámica |

---

## Salidas

Al ejecutarse, el script produce:

- Convergencia de $T_\text{res}$ por iteración (gráfica semilogarítmica)
- Morfología arbórea 3D en el punto estacionario
- Descomposición $T_\text{geo}$ / $T_\text{hiper}$ con valores numéricos
- Espectro singular del Jacobiano y número de condición $\kappa$
- Topografía del funcional $\Phi(\alpha_1, \alpha_2)$ con trayectoria dinámica
- Resumen de verificación MCFT

---

## Contexto teórico

La geometría óptima $\boldsymbol{\alpha}^*$ minimiza el funcional:

$$\Phi(\boldsymbol{\alpha}) = \tfrac{1}{2}\|G(\boldsymbol{\alpha})\|^2$$

donde $G(\boldsymbol{\alpha}) = \sum_k (\mathbf{I} - \mathbf{u}_k \mathbf{u}_k^\top)\mathbf{F}_k$ es la suma de las fuerzas internas proyectadas ortogonalmente al eje de cada rama. En el punto estacionario, la componente $T_\text{geo}$ se anula y el residuo residual $T_\text{hiper}$ es irreducible por variaciones angulares — consecuencia directa de que $J_G \in \mathbb{R}^{3 \times 2}$ tiene rango a lo más 2.

---

## 📖 Referencias

- Timoshenko, S. P., & Gere, J. M. (2000). *Theory of Elastic Stability*
- Ochsendorf, J. A. (2010). *Guastavino Vaulting*
- Block, P., & Ochsendorf, J. (2007). Thrust Network Analysis

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo `LICENSE` para detalles.

## 👤 Autor

**Luis Alberto Valencia Pérez**
- GitHub: [@LuisValencia4k](https://github.com/LuisValencia4k)

## ⚠️ Disclaimer

Este código es de naturaleza académica/investigativa. Verificar resultados independientemente antes de aplicaciones de ingeniería críticas.

## 📞 Soporte

Para reportar bugs o sugerir mejoras, abre un [Issue](https://github.com/LuisValencia4k/form-finding-arboreal-3D-2Ramas/issues).

---

**Última actualización**: Junio 2026
