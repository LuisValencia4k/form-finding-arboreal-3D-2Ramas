# Form-Finding Arboreal 3D con 2 Ramas

> Motor de análisis estructural para optimización de geometría de estructuras arbóreas tridimensionales mediante form-finding funicular y métodos de elementos finitos.

## 📋 Descripción

Este proyecto implementa un motor computacional especializado para el análisis y optimización de estructuras arbóreas 3D con dos ramas principales. Combina técnicas avanzadas de form-finding funicular con análisis de elementos finitos para determinar geometrías óptimas que minimizan deformaciones y tensiones bajo cargas gravitacionales.

### Características principales

- **Form-Finding Funicular**: Optimización de geometría basada en principios de funiculares para estructuras arbóreas
- **FEM de Vigas Timoshenko**: Análisis preciso considerando deformación por cortante
- **Relajación Angular Dinámica**: Algoritmo iterativo para convergencia geométrica
- **Análisis Espectral**: Determinación de frecuencias naturales y modos de vibración
- **Verificación Estructural**: Validación de estados límite según normativas
- **Visualización 3D**: Representación interactiva de geometrías y resultados

## 🚀 Inicio Rápido

### Requisitos previos

- Python 3.8+
- NumPy
- SciPy
- [Especificar otras dependencias clave]

### Instalación

```bash
git clone https://github.com/LuisValencia4k/form-finding-arboreal-3D-2Ramas.git
cd form-finding-arboreal-3D-2Ramas
pip install -r requirements.txt
```

### Uso básico

```python
from arboreal_form_finding import TreeStructure

# Crear estructura arbórea
tree = TreeStructure(
    num_branches=2,
    branch_length=10.0,
    material='steel'
)

# Ejecutar análisis de form-finding
tree.optimize_geometry()

# Obtener resultados
results = tree.get_analysis_results()
tree.visualize()
```

## 📚 Documentación

### Conceptos teóricos

- **Form-Finding Funicular**: Encuentra formas donde la estructura trabajaría únicamente en tracción/compresión axial
- **Timoshenko Beam Theory**: Incluye efectos de deformación por cortante, más preciso para vigas cortas y robustas
- **Relajación Angular Dinámica**: Método iterativo que ajusta ángulos en nodos para minimizar energía de deformación

### Estructura del código

```
form-finding-arboreal-3D-2Ramas/
├── src/
│   ├── core/              # Núcleo del motor
│   ├── fem/               # Módulo de elementos finitos
│   ├── optimization/      # Algoritmos de optimización
│   └── utils/             # Utilidades
├── tests/                 # Suite de pruebas
├── examples/              # Ejemplos de uso
├── docs/                  # Documentación técnica
└── requirements.txt       # Dependencias
```

## 🔧 Configuración

### Parámetros principales

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `num_branches` | int | Número de ramas (fijo en 2) |
| `branch_length` | float | Longitud de ramas en metros |
| `material` | str | Material estructural |
| `max_iterations` | int | Iteraciones máximas de optimización |
| `tolerance` | float | Tolerancia de convergencia |

## 📊 Resultados y Análisis

El proyecto genera múltiples tipos de análisis:

- **Análisis geométrico**: Optimización de coordenadas de nodos
- **Análisis de esfuerzos**: Distribución de tensiones en elementos
- **Análisis dinámico**: Modos y frecuencias de vibración
- **Verificación**: Estados límite de servicio y últimos

## 🧪 Testing

```bash
# Ejecutar suite completa de pruebas
pytest tests/

# Con reporte de cobertura
pytest --cov=src tests/
```

## 📈 Ejemplos

Consulta la carpeta `examples/` para casos de uso:

- `example_basic_analysis.py`: Análisis básico de estructura de 2 ramas
- `example_optimization.py`: Optimización completa con verificación
- `example_visualization.py`: Generación de visualizaciones 3D

## 🔍 Validación

El código incluye verificación contra:

- Equilibrio de fuerzas nodales
- Cierre de ecuaciones de compatibilidad
- Cumplimiento de límites de tensión

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

**Luis Valencia**
- GitHub: [@LuisValencia4k](https://github.com/LuisValencia4k)

## ⚠️ Disclaimer

Este código es de naturaleza académica/investigativa. Verificar resultados independientemente antes de aplicaciones de ingeniería críticas.

## 📞 Soporte

Para reportar bugs o sugerir mejoras, abre un [Issue](https://github.com/LuisValencia4k/form-finding-arboreal-3D-2Ramas/issues).

---

**Última actualización**: Junio 2026
