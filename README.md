# Navegación Autónoma basada en Deep Q-Network (DQN) sobre Simulador Stage

Este repositorio contiene la implementación de un sistema de navegación autónoma reactiva ("mapless") utilizando el algoritmo Deep Q-Network (DQN) para un robot móvil diferencial simulado en Stage. El proyecto fue desarrollado sobre **ROS 2 Jazzy** en **Ubuntu 24.04**.

El objetivo principal es que el agente aprenda a navegar hacia objetivos generados aleatoriamente evitando obstáculos, utilizando únicamente información local (LiDAR y Odometría), sin disponer de un mapa global para la planificación de trayectorias.

El sistema integra un wrapper personalizado de odometría (`OdomResetWrapper`) para gestionar la sincronización de datos en simulaciones aceleradas, mitigando la pérdida de eventos y garantizando un entrenamiento estable.

## Requisitos del Sistema

* **Sistema Operativo:** Ubuntu 24.04 LTS (Noble Numbat)
* **Middleware:** ROS 2 Jazzy Jalisco
* **Simulador:** Stage (paquete `stage_ros2`)

## Instalación y Dependencias

### 1. Paquetes de ROS 2
Es necesario instalar los paquetes de soporte para Stage y los mensajes estándar de navegación:

```bash
sudo apt update
sudo apt install \
  ros-jazzy-stage-ros \
  ros-jazzy-navigation-msgs \
  ros-jazzy-geometry-msgs \
  ros-jazzy-sensor-msgs \
  ros-jazzy-nav-msgs \
  ros-jazzy-std-srvs \
  ros-jazzy-tf2-ros
```

### 2. Librerías de Python

El agente DQN utiliza scikit-learn para la aproximación de la función Q. Adicionalmente, se requieren librerías para el procesamiento de imágenes (validación de objetivos) y visualización:

```bash
sudo apt install python3-sklearn python3-numpy python3-matplotlib python3-opencv
```

## Configuración del Espacio de Trabajo

1. Crear el workspace y clonar el repositorio:

```bash
mkdir -p ~/project_rmov/src
cd ~/project_rmov/src
git clone https://github.com/dannycsa/final_project_7.git .
```

2. Compilar los paquetes:

```bash
cd ~/project_rmov
colcon build --packages-select dqn_project stage_ros2
source install/setup.bash
```

## Ejecución del Proyecto

El sistema requiere dos terminales: una para el simulador y otra para el nodo de control (Entrenamiento o Test).

### Paso 1: Iniciar el Simulador (Terminal 1)

En ambas modalidades (Train o Test), primero se debe levantar el entorno de simulación con el mundo "Cave".

```bash
cd ~/project_rmov
source install/setup.bash
ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false
```

### Paso 2: Ejecutar el Agente (Terminal 2)

#### Opción A: Entrenamiento (Training Node)

Este modo inicia el proceso de aprendizaje por refuerzo. El agente explorará el mapa completo (0-12m) y guardará los modelos en `~/dqn_results/`.

```bash
cd ~/project_rmov
source install/setup.bash
ros2 run dqn_project train_node
```

#### Opción B: Validación y Prueba (Test Node)

Este modo carga el modelo final entrenado y evalúa su desempeño en un rango operativo controlado, con visualización en tiempo real.

```bash
cd ~/project_rmov
source install/setup.bash
# Asegúrate de usar la ruta absoluta a tu modelo
ros2 run dqn_project test_node /home/danny/project_rmov/assets/model_final.pkl
```

**Nota:** Al ejecutar el test, se abrirá una ventana de Matplotlib mostrando el mapa de ocupación, la posición del robot y la validación del objetivo actual.

## Especificaciones Técnicas

### Espacio de Estados (50 dimensiones)

El vector de estado se normaliza entre [0, 1] y se compone de:

* **0-47:** Distancias mínimas de LiDAR discretizadas en 48 sectores angulares.
* **48:** Distancia euclidiana al objetivo.
* **49:** Ángulo relativo hacia el objetivo.

### Espacio de Acciones

El agente opera en un espacio de acciones discreto:

| Índice | Acción | Velocidad Lineal (v) | Velocidad Angular (ω) |
|--------|--------|----------------------|-----------------------|
| 0 | Avance | 0.15 m/s | 0.0 rad/s |
| 1 | Giro Izquierda | 0.00 m/s | 0.25 rad/s |
| 2 | Giro Derecha | 0.00 m/s | -0.25 rad/s |
| 3 | Curva Izquierda | 0.08 m/s | 0.03 rad/s |
| 4 | Curva Derecha | 0.08 m/s | -0.03 rad/s |

## Estructura del Repositorio

```
final_project_7/
├── assets/                 # Recursos estáticos
│   ├── model_final.pkl     # Modelo entrenado (55.2% Success Rate)
│   ├── solid_cave.png      # Mapa de bits para validación de objetivos
│   └── training_results.png # Gráficas de rendimiento
├── src/
│   ├── dqn_project/        # Paquete principal del agente
│   │   ├── dqn_agent.py       # Implementación del agente (MLPRegressor)
│   │   ├── environment.py     # Interfaz con ROS 2 y Stage
│   │   ├── state_processor.py # Preprocesamiento de sensores
│   │   ├── train_node.py      # Lógica de entrenamiento
│   │   └── test_node.py       # Lógica de validación con visualización
│   └── stage_ros2/         # Paquete del simulador Stage
└── README.md
```

## Autores

Diplomado en Robótica - Universidad Católica Boliviana "San Pablo" (La Paz, Bolivia).

* Alvaro Daniel Callata Suxo
* Jhoselin Adelaida Fernandez Chura
* Patricio Sebastian Flores Criales
