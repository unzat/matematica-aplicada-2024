# Proyecto Python: Instrucciones

Este proyecto en Python tiene como objetivo realizar un conjunto de operaciones y generar resultados tanto en consola como en archivos. A continuación se detallan los pasos necesarios para configurar y ejecutar el proyecto.

## Requisitos previos

- Tener Python instalado en tu sistema. (Se recomienda la versión 3.7 o superior).
- Tener `pip` instalado (gestor de paquetes de Python).

## Pasos para ejecutar el proyecto

### 1. Crear un entorno virtual (venv)

python -m venv venv
Este comando creará una carpeta llamada venv en el directorio actual que contiene el entorno virtual.

### 1. Activar el entorno virtual
Una vez creado el entorno virtual, necesitas activarlo.

- En Windows:
.\venv\Scripts\activate

- En macOS o Linux:
source venv/bin/activate

Verás que el prompt de la terminal cambia y ahora debería mostrar el nombre del entorno virtual (generalmente (venv)).

### 3. Instalar las dependencias
Con el entorno virtual activado, ahora puedes instalar las dependencias necesarias para ejecutar el proyecto. Si el proyecto tiene un archivo requirements.txt, puedes instalar todas las librerías listadas allí con el siguiente comando:

pip install -r requirements.txt
Este comando instalará todas las dependencias necesarias para el funcionamiento del proyecto.

### 4. Ejecutar el script principal
Con las dependencias instaladas, ahora puedes ejecutar el archivo principal del proyecto, que en este caso es main.py. Para hacerlo, simplemente usa el siguiente comando:

python main.py
Este comando ejecutará el código contenido en main.py.

### 5. Ver los resultados
Resultados en la consola: Durante la ejecución del script, podrás ver los resultados generales impresos directamente en la consola.

Detalles en un archivo: Además, los detalles específicos de la ejecución se guardarán en un archivo dentro del directorio ./Resultados/. Puedes consultar los archivos allí generados para obtener un desglose más detallado de la ejecución.
