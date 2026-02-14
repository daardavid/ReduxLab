# data_loader_module.py
"""
Módulo para carga y transformación de datos desde archivos Excel, CSV y Parquet.

Este módulo proporciona funciones para:
- Cargar archivos Excel con múltiples hojas (.xlsx, .xls)
- Cargar archivos CSV (.csv) 
- Cargar archivos Parquet (.parquet)
- Transformar datos de formato ancho a largo
- Preparar datos para análisis PCA
- Consolidar datos por país/unidad
- Manejar diferentes estructuras de datos (serie temporal, corte transversal, panel)

Autor: David Armando Abreu Rosique
Fecha: 2025
"""
import pandas as pd
import numpy as np
from functools import reduce
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any

# Importar sistema de logging
from backend.logging_config import get_logger
from backend.error_recovery import recoverable

# Importar módulos de seguridad
from backend.security_utils import validate_file_path, SecurityError
from backend.secure_error_handler import handle_file_operation_error, ProcessingError, safe_exception_handler

logger = get_logger("data_loader_module")


@recoverable
@safe_exception_handler
def load_excel_file(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Carga un archivo Excel y retorna todas sus hojas como DataFrames.

    Esta función maneja errores de manera robusta y proporciona información
    detallada sobre cualquier problema encontrado durante la carga.

    Args:
        file_path (str): Ruta completa al archivo Excel (.xlsx, .xls)

    Returns:
        Optional[Dict[str, pd.DataFrame]]: Diccionario donde las claves son los nombres
        de las hojas y los valores son DataFrames correspondientes. Retorna None si
        hay errores críticos en la carga del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada
        PermissionError: Si no hay permisos para leer el archivo

    Example:
        >>> data = load_excel_file("datos_socioeconomicos.xlsx")
        >>> if data:
        ...     print(f"Hojas cargadas: {list(data.keys())}")
        ...     print(f"Forma de la primera hoja: {list(data.values())[0].shape}")

    Note:
        - El archivo debe tener al menos una hoja válida
        - Se imprime información de progreso durante la carga
        - Los errores se registran con traceback para debugging
    """
    try:
        # Validate file path securely
        try:
            validated_path = validate_file_path(file_path, allowed_extensions=['.xlsx', '.xls'])
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "security validation")
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "path validation")
            logger.error(error_msg)
            return None

        try:
            excel_data = pd.ExcelFile(str(validated_path))
            sheet_names = excel_data.sheet_names
            logger.info("Cargando hojas del archivo: %s", validated_path.name)
        except Exception as e_open:
            error_msg = handle_file_operation_error(e_open, str(validated_path), "opening Excel file")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None

        dataframes = {}
        if not sheet_names:
            logger.warning("El archivo Excel no contiene hojas.")
            return {}

        for sheet_name in sheet_names:
            try:
                df = excel_data.parse(sheet_name)
                # Validate DataFrame doesn't contain malicious content
                if _validate_dataframe_security(df, 'excel'):
                    dataframes[sheet_name] = df
                else:
                    logger.warning(f"Sheet '{sheet_name}' failed security validation and was skipped")
            except Exception as e_parse:
                error_msg = handle_file_operation_error(e_parse, str(validated_path), f"parsing sheet '{sheet_name}'")
                logger.error(error_msg)
                logger.error(traceback.format_exc())

        if not dataframes:
            logger.warning("No se pudo parsear ninguna hoja de datos valida del archivo.")
        return dataframes
    except FileNotFoundError:
        error_msg = handle_file_operation_error(FileNotFoundError("File not found"), file_path, "loading")
        logger.error(error_msg)
        return None
    except Exception as e_load:
        error_msg = handle_file_operation_error(e_load, file_path, "processing Excel file")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None


@recoverable
@safe_exception_handler
def load_parquet_file(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Carga un archivo Parquet y retorna su contenido como DataFrame(s).

    Args:
        file_path (str): Ruta completa al archivo Parquet (.parquet)

    Returns:
        Optional[Dict[str, pd.DataFrame]]: Diccionario donde las claves son los nombres
        de las tablas y los valores son DataFrames correspondientes. Para archivos parquet
        estándar, se retorna un solo DataFrame con clave 'data'. Retorna None si hay
        errores críticos en la carga del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada
        PermissionError: Si no hay permisos para leer el archivo

    Example:
        >>> data = load_parquet_file("datos_socioeconomicos.parquet")
        >>> if data:
        ...     print(f"Tablas cargadas: {list(data.keys())}")
        ...     print(f"Forma del DataFrame: {data['data'].shape}")
    """
    try:
        # Validate file path securely
        try:
            validated_path = validate_file_path(file_path, allowed_extensions=['.parquet'])
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "security validation")
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "path validation")
            logger.error(error_msg)
            return None

        try:
            logger.info("Cargando archivo Parquet: %s", validated_path.name)
            
            # Try to load the parquet file
            # Parquet files typically contain a single DataFrame, but we maintain
            # the same interface as load_excel_file for compatibility
            df = pd.read_parquet(str(validated_path))
            
            # Validate DataFrame doesn't contain malicious content
            if not _validate_dataframe_security(df):
                logger.warning("Parquet file failed security validation")
                return None
                
            # Return in the same format as Excel files (dict of sheet_name -> DataFrame)
            # For parquet, we use 'data' as the default sheet name
            dataframes = {'data': df}
            
            logger.info("Archivo Parquet cargado exitosamente. Forma: %s", df.shape)
            return dataframes
            
        except Exception as e_load:
            error_msg = handle_file_operation_error(e_load, str(validated_path), "loading Parquet file")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None
            
    except FileNotFoundError:
        error_msg = handle_file_operation_error(FileNotFoundError("File not found"), file_path, "loading")
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = handle_file_operation_error(e, file_path, "processing Parquet file")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None


@recoverable
@safe_exception_handler
def load_any_file(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Carga un archivo de cualquier formato soportado (Excel, CSV, Parquet) 
    y retorna su contenido como DataFrame(s).

    Args:
        file_path (str): Ruta completa al archivo

    Returns:
        Optional[Dict[str, pd.DataFrame]]: Diccionario donde las claves son los nombres
        de las hojas/tablas y los valores son DataFrames correspondientes. Retorna None 
        si hay errores críticos en la carga del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada
        PermissionError: Si no hay permisos para leer el archivo
        ValueError: Si el formato del archivo no está soportado

    Example:
        >>> # Para archivos Excel (.xlsx, .xls)
        >>> data = load_any_file("datos.xlsx")
        >>> # Para archivos CSV (.csv)
        >>> data = load_any_file("datos.csv") 
        >>> # Para archivos Parquet (.parquet)
        >>> data = load_any_file("datos.parquet")
        >>> if data:
        ...     print(f"Datos cargados: {list(data.keys())}")
    """
    try:
        file_path_str = str(file_path).lower()
        
        if file_path_str.endswith(('.xlsx', '.xls')):
            return load_excel_file(file_path)
        elif file_path_str.endswith('.parquet'):
            return load_parquet_file(file_path)
        elif file_path_str.endswith('.csv'):
            return load_csv_file(file_path)
        else:
            error_msg = f"Formato de archivo no soportado: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        error_msg = handle_file_operation_error(e, file_path, "loading any file")
        logger.error(error_msg)
        return None


@recoverable
@safe_exception_handler
def load_csv_file(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Carga un archivo CSV y retorna su contenido como DataFrame.

    Args:
        file_path (str): Ruta completa al archivo CSV (.csv)

    Returns:
        Optional[Dict[str, pd.DataFrame]]: Diccionario donde la clave es 'data'
        y el valor es el DataFrame correspondiente. Retorna None si hay errores 
        críticos en la carga del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta especificada
        PermissionError: Si no hay permisos para leer el archivo

    Example:
        >>> data = load_csv_file("datos_socioeconomicos.csv")
        >>> if data:
        ...     print(f"DataFrame cargado. Forma: {data['data'].shape}")
    """
    try:
        # Validate file path securely
        try:
            validated_path = validate_file_path(file_path, allowed_extensions=['.csv'])
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "security validation")
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "path validation")
            logger.error(error_msg)
            return None

        try:
            logger.info("Cargando archivo CSV: %s", validated_path.name)
            
            # Try different encodings for CSV files
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(str(validated_path), encoding=encoding)
                    logger.info(f"CSV file loaded successfully with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError("Could not read CSV file with any of the attempted encodings")
            
            # Validate DataFrame doesn't contain malicious content
            if not _validate_dataframe_security(df):
                logger.warning("CSV file failed security validation")
                return None
                
            # Return in the same format as Excel files (dict of sheet_name -> DataFrame)
            dataframes = {'data': df}
            
            logger.info("Archivo CSV cargado exitosamente. Forma: %s", df.shape)
            return dataframes
            
        except Exception as e_load:
            error_msg = handle_file_operation_error(e_load, str(validated_path), "loading CSV file")
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None
            
    except FileNotFoundError:
        error_msg = handle_file_operation_error(FileNotFoundError("File not found"), file_path, "loading")
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = handle_file_operation_error(e, file_path, "processing CSV file")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None


@recoverable
@safe_exception_handler
def load_correlation_data_parquet(file_path: str, unit_col: str = 'Unit', year_col: str = 'Year',
                                 alt_unit_cols: List[str] = None, alt_year_cols: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Load correlation data from Parquet file for analysis.

    This function handles Parquet files for correlation analysis where:
    - The file contains data ready for correlation analysis
    - Expected format: [Unit, Year, Indicator1, Indicator2, ...]
    
    Args:
        file_path (str): Path to Parquet file
        unit_col (str): Name of the unit column (default: 'Unit')
        year_col (str): Name of the year column (default: 'Year')
        alt_unit_cols (List[str]): Alternative names for unit column
        alt_year_cols (List[str]): Alternative names for year column

    Returns:
        Optional[pd.DataFrame]: DataFrame with the loaded data, or None if loading fails

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid
    """
    try:
        # Validate file path securely
        try:
            validated_path = validate_file_path(file_path, allowed_extensions=['.parquet'])
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "security validation")
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = handle_file_operation_error(e, file_path, "path validation")
            logger.error(error_msg)
            return None

        logger.info(f"Loading correlation data from Parquet: {validated_path.name}")

        # Load the parquet file
        df = pd.read_parquet(str(validated_path))
        
        # Validate DataFrame doesn't contain malicious content
        if not _validate_dataframe_security(df):
            logger.warning("Parquet correlation file failed security validation")
            return None

        # Validate required columns with fallback alternatives
        required_cols = [unit_col, year_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        # Try alternative column names if defaults are missing
        if missing_cols:
            logger.info(f"Default columns {missing_cols} not found, trying alternatives...")

            # Alternative column names
            alt_unit_cols = alt_unit_cols or ['Empresa', 'Company', 'Unidad', 'Unit']
            alt_year_cols = alt_year_cols or ['Año', 'Year', 'Ano']

            # Try to map missing columns
            column_mapping = {}

            if unit_col not in df.columns:
                for alt_col in alt_unit_cols:
                    if alt_col in df.columns:
                        column_mapping[alt_col] = unit_col
                        logger.info(f"Mapped '{alt_col}' to '{unit_col}'")
                        break

            if year_col not in df.columns:
                for alt_col in alt_year_cols:
                    if alt_col in df.columns:
                        column_mapping[alt_col] = year_col
                        logger.info(f"Mapped '{alt_col}' to '{year_col}'")
                        break

            # Apply column mapping
            if column_mapping:
                logger.info(f"Before rename - columns: {df.columns.tolist()[:5]}")
                df = df.rename(columns=column_mapping)
                logger.info(f"Applied column mapping: {column_mapping}")
                logger.info(f"After rename - columns: {df.columns.tolist()[:5]}")

            # Check again after mapping
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                available_cols = df.columns.tolist()[:10]
                logger.error(f"Still missing after mapping: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {available_cols}...")

        # Check for indicator columns
        indicator_cols = [col for col in df.columns if col not in required_cols]
        if not indicator_cols:
            raise ValueError("No indicator columns found. Data must have columns beyond Unit and Year.")

        # Data validation and cleaning
        df = df.copy()

        # Convert year column to numeric
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

        # Remove rows with invalid years
        invalid_years = df[year_col].isna().sum()
        if invalid_years > 0:
            logger.warning(f"Removing {invalid_years} rows with invalid year values")
            df = df.dropna(subset=[year_col])

        # Convert year to integer if possible
        try:
            df[year_col] = df[year_col].astype(int)
        except (ValueError, TypeError):
            logger.warning("Could not convert year column to integer")

        # Convert indicator columns to numeric
        for col in indicator_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check data quality
        total_rows = len(df)
        complete_rows = df.dropna().shape[0]
        completeness = complete_rows / total_rows if total_rows > 0 else 0

        logger.info(f"Parquet correlation data loaded successfully:")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Units: {df[unit_col].nunique()}")
        logger.info(f"  - Years: {df[year_col].nunique()} (range: {df[year_col].min()} - {df[year_col].max()})")
        logger.info(f"  - Indicators: {len(indicator_cols)}")
        logger.info(f"  - Data completeness: {completeness:.1%}")

        if completeness < 0.1:
            logger.warning("Warning: Very low data completeness. Consider data preprocessing.")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading Parquet correlation data from {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def prompt_select_sheets(available_sheet_names):
    """
    Permite al usuario seleccionar hojas (indicadores) de una lista.
    Ahora incluye la opción de escribir 'TODOS' para seleccionar todas las hojas.

    Args:
        available_sheet_names (list): Lista de nombres de hojas disponibles.

    Returns:
        list: Lista de nombres de hojas seleccionadas. Vacía si no se selecciona ninguna.
    """
    if not available_sheet_names:
        print("No hay hojas disponibles para seleccionar.")
        return []

    print("\n--- Hojas (Indicadores) Disponibles para Selección ---")
    for i, sheet_name in enumerate(available_sheet_names):
        print(f"  {i+1}. {sheet_name}")

    selected_names_final = []
    while True:
        # Mensaje del prompt para el usuario
        prompt_message = (
            "Ingresa los números de las hojas/indicadores que quieres usar, separados por comas (ej. 1,3),\n"
            "escribe 'TODOS' para seleccionar todas, o deja vacío para no seleccionar ninguna: "
        )
        selection_str = input(prompt_message)

        # Comprobar si el usuario escribió 'TODOS' (insensible a mayúsculas/minúsculas)
        if selection_str.strip().lower() == "todos":
            print("\n--- Todas las Hojas/Indicadores Seleccionados ---")
            # No es necesario imprimir todos aquí, main.py ya lo hace con los seleccionados.
            # Pero si quieres una confirmación inmediata:
            # for name in available_sheet_names:
            #     print(f"  - {name}")
            return (
                available_sheet_names  # Devuelve la lista completa de nombres de hojas
            )

        # Comprobar si la entrada está vacía
        if not selection_str.strip():
            print("No se seleccionó ninguna hoja.")
            return []

        try:
            selected_indices = [
                int(idx.strip()) - 1 for idx in selection_str.split(",")
            ]
            temp_selected_names = []
            valid_selection_made = False
            for i in selected_indices:
                if 0 <= i < len(available_sheet_names):
                    temp_selected_names.append(available_sheet_names[i])
                    valid_selection_made = True
                else:
                    print(
                        f"Advertencia: El número {i+1} está fuera de rango y será ignorado."
                    )

            if (
                not valid_selection_made and temp_selected_names
            ):  # Si solo hubo inválidos pero se intentó algo
                print(
                    "Todos los números ingresados estaban fuera de rango. Intenta de nuevo."
                )
                continue

            # Eliminar duplicados manteniendo el orden de la primera aparición
            seen = set()
            selected_names_final = [
                x for x in temp_selected_names if not (x in seen or seen.add(x))
            ]

            if selected_names_final:
                return selected_names_final
            else:
                print(
                    "No se seleccionó ninguna hoja válida con los números ingresados. Intenta de nuevo."
                )

        except ValueError:
            # Mensaje de error actualizado
            print(
                "Error: Entrada inválida. Ingresa solo números separados por comas (ej. 1,3), la palabra 'TODOS', o deja vacío. Intenta de nuevo."
            )


def transformar_df_indicador_v1(
    df_original,
    col_paises_nombre_original="Unnamed: 0",
    nuevo_nombre_indice_paises="Pais",
):
    logger.info("Transformando DataFrame (Estructura V1)")
    if df_original is None or df_original.empty:
        logger.warning("DataFrame original esta vacio. No se puede transformar.")
        return None
    try:
        df = df_original.copy()
        if col_paises_nombre_original not in df.columns:
            logger.error(
                "La columna de paises '%s' no se encuentra en el DataFrame.",
                col_paises_nombre_original,
            )
            logger.error("Columnas disponibles: %s", df.columns.tolist())
            return None
        df.set_index(col_paises_nombre_original, inplace=True)
        df.index.name = nuevo_nombre_indice_paises
        logger.info(
            "Indice establecido a '%s'. Columnas actuales (anios): %s",
            df.index.name,
            df.columns.tolist(),
        )
        logger.info("Transponiendo DataFrame...")
        df_transformado = df.transpose()
        df_transformado.index.name = "Año"
        df_transformado.index = pd.to_numeric(df_transformado.index, errors="coerce")
        original_rows = len(df_transformado)
        df_transformado.dropna(axis=0, how="all", subset=None, inplace=True)
        df_transformado = df_transformado[df_transformado.index.notna()]
        if len(df_transformado) < original_rows:
            logger.info(
                "Se eliminaron %d filas con Anios no validos o completamente vacias.",
                original_rows - len(df_transformado),
            )
        if df_transformado.empty:
            logger.warning("DataFrame vacio despues de eliminar Anios no validos.")
            return None
        try:
            df_transformado.index = df_transformado.index.astype(int)
        except ValueError:
            logger.warning("El indice de Anios no pudo ser convertido a entero.")
        logger.info("Convirtiendo valores de datos a numerico...")
        # BUG FIX: Conversión más robusta a numérico para evitar TypeError
        for col_pais in df_transformado.columns:
            try:
                # Convertir cada columna a numérico individualmente
                # Usar apply con una lambda para manejar cada valor individualmente
                df_transformado[col_pais] = df_transformado[col_pais].apply(
                    lambda x: pd.to_numeric(x, errors='coerce') if pd.notna(x) else np.nan
                )
            except Exception as col_error:
                logger.warning("Error convirtiendo columna '%s' a numerico: %s", col_pais, col_error)
                # Intentar conversión alternativa
                try:
                    df_transformado[col_pais] = pd.to_numeric(df_transformado[col_pais], errors='coerce')
                except Exception as fallback_error:
                    logger.error("Error critico en columna '%s': %s. Columna omitida.", col_pais, fallback_error)
                    df_transformado[col_pais] = np.nan
        
        df_transformado.dropna(axis=1, how="all", inplace=True)
        logger.info("Transformacion V1 completada.")
        return df_transformado
    except Exception as e:
        logger.error("Error durante la transformacion del DataFrame (V1): %s", e, exc_info=True)
        return None


def prompt_select_country(data_transformada_indicadores):
    """
    Permite al usuario seleccionar un país de los disponibles en los DataFrames transformados.

    Args:
        data_transformada_indicadores (dict): Diccionario donde las claves son nombres de indicadores
                                              y los valores son DataFrames con Años como índice
                                              y Países como columnas.
    Returns:
        str: El nombre del país seleccionado, o None si no se selecciona ninguno o hay error.
    """
    if not data_transformada_indicadores:
        print("No hay datos transformados disponibles para seleccionar un país.")
        return None

    # Tomar el primer DataFrame del diccionario para obtener la lista de países (columnas)
    # Asumimos que todos los DataFrames transformados tienen una estructura de columnas similar (países)
    primer_nombre_indicador = next(iter(data_transformada_indicadores))
    df_referencia_paises = data_transformada_indicadores[primer_nombre_indicador]

    if df_referencia_paises is None or df_referencia_paises.empty:
        print(
            f"El DataFrame de referencia ('{primer_nombre_indicador}') para listar países está vacío o no existe."
        )
        return None

    paises_disponibles = df_referencia_paises.columns.tolist()

    if not paises_disponibles:
        print("No se encontraron países (columnas) en el DataFrame de referencia.")
        return None

    print("\n--- Países Disponibles para Selección ---")
    for i, pais_nombre in enumerate(paises_disponibles):
        print(f"  {i+1}. {pais_nombre}")

    pais_seleccionado_final = None
    while True:  # Bucle hasta obtener una selección válida o vacía
        selection_str = input(
            "Ingresa el número del país que quieres analizar, o deja vacío para no seleccionar: "
        )

        if not selection_str.strip():
            print("No se seleccionó ningún país.")
            return None

        try:
            selected_index = int(selection_str.strip()) - 1

            if 0 <= selected_index < len(paises_disponibles):
                pais_seleccionado_final = paises_disponibles[selected_index]
                print(f"\n--- País Seleccionado para Análisis ---")
                print(f"  - {pais_seleccionado_final}")
                return pais_seleccionado_final
            else:
                print(
                    f"Advertencia: El número {selected_index + 1} está fuera de rango. Intenta de nuevo."
                )

        except ValueError:
            print("Error: Entrada inválida. Ingresa solo un número. Intenta de nuevo.")


def consolidate_data_for_country(
    data_transformada_indicadores, country_to_analyze, selected_sheet_names
):
    """
    Consolida los datos de múltiples indicadores para un país específico en un solo DataFrame.

    Args:
        data_transformada_indicadores (dict): Diccionario donde las claves son nombres de indicadores
                                              y los valores son DataFrames con Años como índice
                                              y Países como columnas.
        country_to_analyze (str): El nombre del país seleccionado.
        selected_sheet_names (list): Lista de los nombres de las hojas/indicadores seleccionados,
                                     para mantener el orden de las columnas y los nombres.
    Returns:
        pd.DataFrame: Un DataFrame con Años como índice y los indicadores seleccionados como columnas
                      para el país especificado. Devuelve un DataFrame vacío si hay error.
    """
    if not data_transformada_indicadores:
        logger.warning("No hay datos transformados de indicadores para consolidar.")
        return pd.DataFrame()
    if not country_to_analyze:
        logger.warning("No se especifico un pais para la consolidacion.")
        return pd.DataFrame()
    if not selected_sheet_names:
        logger.warning("No se especificaron nombres de hojas/indicadores para la consolidacion.")
        return pd.DataFrame()

    logger.info("Consolidando datos para el pais: %s", country_to_analyze)
    logger.info("Indicadores solicitados: %s", selected_sheet_names)

    lista_series_del_pais = []
    indicadores_procesados_exitosamente = []

    for nombre_indicador in selected_sheet_names:
        if nombre_indicador in data_transformada_indicadores:
            df_indicador_actual = data_transformada_indicadores[nombre_indicador]

            if df_indicador_actual is None or df_indicador_actual.empty:
                logger.warning(
                    "El DataFrame transformado para el indicador '%s' esta vacio o es None. Se omitira.",
                    nombre_indicador,
                )
                continue  # Saltar al siguiente indicador

            if country_to_analyze in df_indicador_actual.columns:
                # Seleccionar la serie de datos para el país
                serie_pais_indicador = df_indicador_actual[country_to_analyze].copy()

                # Verificar que la serie tenga al menos algunos datos válidos
                datos_validos = serie_pais_indicador.dropna()
                if len(datos_validos) == 0:
                    logger.warning(
                        "El indicador '%s' para '%s' no tiene datos validos (todos son NaN). Se omitira.",
                        nombre_indicador,
                        country_to_analyze,
                    )
                    continue

                # Renombrar la Serie para que su nombre sea el del indicador (nombre de la hoja)
                # Esto será el nombre de la columna en el DataFrame final
                serie_pais_indicador.name = nombre_indicador
                lista_series_del_pais.append(serie_pais_indicador)
                indicadores_procesados_exitosamente.append(nombre_indicador)
                logger.info(
                    "Datos del indicador '%s' para '%s' aniadidos. (%d valores validos)",
                    nombre_indicador,
                    country_to_analyze,
                    len(datos_validos),
                )
            else:
                logger.warning(
                    "El pais '%s' no se encontro como columna en el indicador transformado '%s'. Se omitira.",
                    country_to_analyze,
                    nombre_indicador,
                )
                paises_disponibles = list(df_indicador_actual.columns)[
                    :10
                ]  # Mostrar solo los primeros 10
                logger.warning(
                    "Paises disponibles en este indicador: %s%s",
                    paises_disponibles,
                    "..." if len(df_indicador_actual.columns) > 10 else "",
                )
        else:
            logger.warning(
                "El indicador '%s' no se encontro en los datos transformados. Se omitira.",
                nombre_indicador,
            )

    if not lista_series_del_pais:
        logger.warning(
            "No se pudieron obtener datos para ningun indicador para el pais '%s'. "
            "Posibles causas: el pais no existe en los indicadores seleccionados, "
            "los nombres de pais no coinciden entre indicadores, o los indicadores "
            "seleccionados no tienen datos para este pais.",
            country_to_analyze,
        )
        return pd.DataFrame()

    logger.info("Indicadores procesados exitosamente: %s", indicadores_procesados_exitosamente)

    # Concatenar todas las Series en un solo DataFrame.
    # Cada Serie se convertirá en una columna. El índice (Año) se alineará automáticamente.
    try:
        df_consolidado_final = pd.concat(lista_series_del_pais, axis=1)

        # Verificaciones adicionales de calidad
        total_valores = df_consolidado_final.size
        valores_nan = df_consolidado_final.isnull().sum().sum()
        porcentaje_nan = (valores_nan / total_valores * 100) if total_valores > 0 else 0

        logger.info("Datos Consolidados para %s (listos para ACP)", country_to_analyze)
        logger.info("Forma final: %s", df_consolidado_final.shape)
        logger.info(
            "Rango de anios: %s - %s",
            df_consolidado_final.index.min(),
            df_consolidado_final.index.max(),
        )
        logger.info(
            "Valores faltantes: %d/%d (%.1f%%)",
            valores_nan,
            total_valores,
            porcentaje_nan,
        )

        return df_consolidado_final
    except Exception as e_concat:
        logger.error(
            "Error al concatenar las series de datos para el pais '%s': %s",
            country_to_analyze,
            e_concat,
            exc_info=True,
        )
        return pd.DataFrame()


# En data_loader_module.py (o adaptado para main.py)
def preparar_datos_corte_transversal(
    all_sheets_data: Dict[str, pd.DataFrame],
    selected_indicators_codes: List[str],
    selected_countries_names: List[str],
    target_year: Union[int, str],
    col_paises_nombre_original: str = "Unnamed: 0",
) -> pd.DataFrame:
    """
    Prepara un DataFrame para análisis de corte transversal para un año específico.

    Esta función extrae datos de múltiples indicadores para países seleccionados
    en un año determinado, creando una matriz adecuada para análisis PCA.

    Args:
        all_sheets_data (Dict[str, pd.DataFrame]): Diccionario con DataFrames por indicador
        selected_indicators_codes (List[str]): Lista de códigos de indicadores a incluir
        selected_countries_names (List[str]): Lista de códigos de países a incluir
        target_year (Union[int, str]): Año objetivo para el análisis
        col_paises_nombre_original (str, optional): Nombre de la columna que contiene
            los códigos de países. Por defecto 'Unnamed: 0'.

    Returns:
        pd.DataFrame: DataFrame con países como filas e indicadores como columnas,
        conteniendo los valores para el año especificado. Las filas corresponden
        a los países seleccionados y las columnas a los indicadores.

    Raises:
        ValueError: Si no hay datos suficientes para el año especificado
        KeyError: Si algún indicador no existe en los datos

    Example:
        >>> df_cross = preparar_datos_corte_transversal(
        ...     data, ['GDP_growth', 'Inflation'], ['USA', 'MEX'], 2020
        ... )
        >>> print(f"Países: {df_cross.index.tolist()}")
        >>> print(f"Indicadores: {df_cross.columns.tolist()}")

    Note:
        - Los valores faltantes se mantienen como NaN y deben ser manejados posteriormente
        - El DataFrame resultante está listo para estandarización y PCA
        - Se maneja automáticamente la detección de formato de año (int, str, float)
    """
    """
    [CORREGIDA v2] Prepara un DataFrame para un análisis de corte transversal para un año específico.
    Maneja la configuración del índice de forma más robusta para evitar errores.
    """
    logger.info("Preparando datos de corte transversal para el anio: %s", target_year)

    target_year_str = str(target_year)
    target_year_int = int(target_year)

    list_of_series_for_year = []

    for indicator_code in selected_indicators_codes:
        if indicator_code not in all_sheets_data:
            logger.warning("Indicador '%s' no encontrado. Se omitira.", indicator_code)
            continue

        df_indicator = all_sheets_data[indicator_code].copy()

        # --- LÓGICA DE ÍNDICE CORREGIDA Y ROBUSTA ---
        # 1. Comprobar si la columna de países está en las columnas del DataFrame.
        if col_paises_nombre_original in df_indicator.columns:
            # Si está, establecerla como el índice.
            # Prevenir errores por nombres de países duplicados, mantener solo el primero.
            if df_indicator[col_paises_nombre_original].duplicated().any():
                df_indicator = df_indicator.drop_duplicates(
                    subset=[col_paises_nombre_original], keep="first"
                )
            df_indicator.set_index(col_paises_nombre_original, inplace=True)

        # 2. Si no estaba en las columnas, verificar si el índice actual NO es ya el correcto.
        elif df_indicator.index.name != col_paises_nombre_original:
            # Si no está en las columnas y el índice actual tampoco es el correcto,
            # significa que no podemos identificar los países en esta hoja. La omitimos.
            logger.warning(
                "No se encontro la columna de paises '%s' en el indicador '%s'. Se omitira.",
                col_paises_nombre_original,
                indicator_code,
            )
            continue
        # Si el código llega aquí, significa que el índice del DataFrame es correcto.
        # --- FIN DE LA LÓGICA DE ÍNDICE CORREGIDA ---

        # Buscar la columna del año (como número, texto o float)
        year_col_to_use = None
        if target_year_int in df_indicator.columns:
            year_col_to_use = target_year_int
        elif target_year_str in df_indicator.columns:
            year_col_to_use = target_year_str
        elif float(target_year_int) in df_indicator.columns:
            year_col_to_use = float(target_year_int)

        if year_col_to_use is None:
            logger.warning(
                "Anio '%s' no encontrado en indicador '%s'. Se generaran NaNs para este indicador.",
                target_year,
                indicator_code,
            )
            nan_series = pd.Series(
                index=selected_countries_names, name=indicator_code, dtype=float
            )
            list_of_series_for_year.append(nan_series)
            continue

        # Extraer los datos para los países seleccionados usando .reindex()
        series_from_sheet = df_indicator[year_col_to_use]
        indicator_series_for_year = series_from_sheet.reindex(selected_countries_names)
        indicator_series_for_year.name = indicator_code
        list_of_series_for_year.append(indicator_series_for_year)

    if not list_of_series_for_year:
        logger.warning(
            "No se pudieron extraer datos para ningun indicador para el anio %s.",
            target_year,
        )
        return pd.DataFrame(index=selected_countries_names)

    # Concatenar y dar formato final
    df_cross_section = pd.concat(list_of_series_for_year, axis=1)
    df_cross_section = df_cross_section.reindex(selected_countries_names)

    logger.debug("DataFrame para el anio %s (primeras filas):\n%s", target_year, df_cross_section.head())
    return df_cross_section


def preparar_datos_panel_longitudinal(
    all_sheets_data,
    selected_indicators_codes,
    selected_countries_codes,
    col_paises_nombre_original="Unnamed: 0",
):
    """
    [CORREGIDA v2] Prepara un DataFrame en formato de panel (longitudinal).
    Maneja la configuración del índice de forma robusta para evitar errores.
    """
    logger.info("Preparando datos en formato de panel longitudinal")

    panel_data_list = []

    for indicator_code in selected_indicators_codes:
        if indicator_code not in all_sheets_data:
            logger.warning("Indicador '%s' no encontrado. Se omitira.", indicator_code)
            continue

        df_indicator = all_sheets_data[indicator_code].copy()

        # --- LÓGICA CORREGIDA PARA MANEJAR EL ÍNDICE ---
        # Primero, verificamos si la columna de países ('Unnamed: 0') ya es el índice.
        if df_indicator.index.name == col_paises_nombre_original:
            # Si lo es, lo convertimos de nuevo en una columna para poder usar 'melt' después.
            df_indicator.reset_index(inplace=True)

        # Ahora, verificamos si la columna de países existe.
        # Si no existe en este punto, la hoja tiene un formato incorrecto y la omitimos.
        if col_paises_nombre_original not in df_indicator.columns:
            logger.warning(
                "No se encontro la columna de paises '%s' en el indicador '%s'. Se omitira.",
                col_paises_nombre_original,
                indicator_code,
            )
            continue
        # --- FIN DE LA LÓGICA CORREGIDA ---

        # Filtrar solo por los países seleccionados
        df_indicator = df_indicator[
            df_indicator[col_paises_nombre_original].isin(selected_countries_codes)
        ]
        if df_indicator.empty:
            continue

        # Usar pd.melt para convertir de formato ancho a largo
        try:
            df_melted = df_indicator.melt(
                id_vars=[col_paises_nombre_original],
                var_name="Año",
                value_name=indicator_code,
            )
            panel_data_list.append(df_melted)
        except Exception as e_melt:
            logger.error("Error al transformar el indicador '%s': %s", indicator_code, e_melt)

    if not panel_data_list:
        logger.warning("No se pudo procesar ningun indicador en formato de panel.")
        return pd.DataFrame()

    # Unir todos los DataFrames de indicadores en uno solo
    if len(panel_data_list) == 1:
        df_panel_final = panel_data_list[0]
    else:
        df_panel_final = reduce(
            lambda left, right: pd.merge(
                left, right, on=[col_paises_nombre_original, "Año"], how="outer"
            ),
            panel_data_list,
        )

    # Limpiar y ordenar
    df_panel_final.rename(columns={col_paises_nombre_original: "País"}, inplace=True)
    df_panel_final["Año"] = pd.to_numeric(df_panel_final["Año"], errors="coerce")
    df_panel_final.dropna(subset=["Año"], inplace=True)
    df_panel_final["Año"] = df_panel_final["Año"].astype(int)
    df_panel_final.sort_values(by=["País", "Año"], inplace=True)
    df_panel_final.set_index(["País", "Año"], inplace=True)

    logger.debug("Datos de panel construidos (primeras filas):\n%s", df_panel_final.head())

    return df_panel_final


def _validate_dataframe_security(df: pd.DataFrame, file_type: str = 'unknown') -> bool:
    """
    Validate DataFrame for security issues with appropriate limits for different file types.

    Args:
        df: DataFrame to validate
        file_type: Type of file ('parquet', 'excel', 'csv', 'unknown') for appropriate limits

    Returns:
        bool: True if DataFrame passes security checks
    """
    try:
        # Set appropriate limits based on file type
        if file_type.lower() == 'parquet':
            # Parquet files can be much larger due to compression and efficiency
            max_rows = 10000000  # 10M rows for parquet
            max_columns = 10000  # 10K columns for parquet
            max_cell_size = 50000  # 50KB limit per cell (increased for parquet)
        elif file_type.lower() == 'excel':
            # More restrictive limits for Excel
            max_rows = 1000000   # 1M rows for Excel
            max_columns = 1000   # 1K columns for Excel
            max_cell_size = 10000  # 10KB limit per cell
        elif file_type.lower() == 'csv':
            # CSV limits
            max_rows = 1000000   # 1M rows for CSV
            max_columns = 1000   # 1K columns for CSV
            max_cell_size = 10000  # 10KB limit per cell
        else:
            # Default limits for unknown file types
            max_rows = 1000000   # 1M rows default
            max_columns = 1000   # 1K columns default
            max_cell_size = 10000  # 10KB limit per cell default

        # Check for reasonable DataFrame size
        if len(df) > max_rows:
            logger.warning(f"DataFrame too large ({len(df)} rows), rejecting for security. Limit: {max_rows} for {file_type}")
            return False

        if len(df.columns) > max_columns:
            logger.warning(f"DataFrame has too many columns ({len(df.columns)}), rejecting for security. Limit: {max_columns} for {file_type}")
            return False

        # Check for suspicious column names
        suspicious_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        for col in df.columns:
            col_str = str(col).lower()
            for pattern in suspicious_patterns:
                if pattern in col_str:
                    logger.warning(f"Suspicious column name detected: {col}")
                    return False

        # Check for extremely long strings that might contain malicious content
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                for value in sample_values:
                    value_str = str(value)
                    if len(value_str) > max_cell_size:
                        logger.warning(f"Extremely long string value detected in {file_type} file, rejecting for security")
                        return False

        logger.debug(f"DataFrame security validation passed for {file_type}: {len(df)} rows, {len(df.columns)} columns")
        return True

    except Exception as e:
        logger.warning(f"Error during DataFrame security validation: {e}")
        return False


@recoverable
@safe_exception_handler
def load_correlation_data_multisheet(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load correlation data from Excel file with multi-sheet format for correlation analysis.

    This function handles the new multi-sheet format where:
    - Each sheet represents one indicator (sheet name = indicator name)
    - Column A: Research units (companies)
    - Cell B1: Year (single cell)
    - Rows B2, B3, B4, ...: Indicator values for that year

    Args:
        file_path (str): Path to Excel file

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns [Unit, Year, Indicator, Value]
                               ready for correlation analysis, or None if loading fails

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid
    """
    try:
        # Validate file path securely
        try:
            validated_path = validate_file_path(file_path, allowed_extensions=['.xlsx', '.xls'])
        except SecurityError as e:
            error_msg = handle_file_operation_error(e, file_path, "security validation")
            logger.error(error_msg)
            return None

        logger.info(f"Loading multi-sheet correlation data from: {validated_path.name}")

        # Load all sheets (no artificial row limit — correlation data is typically small)
        excel_data = pd.read_excel(str(validated_path), sheet_name=None, header=None)
        if not excel_data:
            raise ValueError("No sheets found in Excel file")

        all_data = []

        for sheet_name, df in excel_data.items():
            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                continue

            try:
                # Extract year from cell B1 (row 0, column 1)
                year = df.iloc[0, 1]
                if pd.isna(year):
                    logger.warning(f"Sheet '{sheet_name}' has no year in B1, skipping")
                    continue

                # Convert year to numeric
                year = pd.to_numeric(year, errors='coerce')
                if pd.isna(year):
                    logger.warning(f"Sheet '{sheet_name}' has invalid year in B1, skipping")
                    continue

                # Extract units from column A (starting from row 1)
                units = df.iloc[1:, 0].values  # Column A, rows 2+

                # Extract values from column B (starting from row 1)
                values = df.iloc[1:, 1].values  # Column B, rows 2+

                # Create DataFrame for this sheet
                sheet_df = pd.DataFrame({
                    'Unit': units,
                    'Year': year,
                    'Indicator': sheet_name,
                    'Value': values
                })

                # Remove rows with missing units or values
                sheet_df = sheet_df.dropna(subset=['Unit', 'Value'])

                if not sheet_df.empty:
                    all_data.append(sheet_df)
                    logger.info(f"Sheet '{sheet_name}' loaded: {len(sheet_df)} observations for year {year}")
                else:
                    logger.warning(f"Sheet '{sheet_name}' has no valid data, skipping")

            except Exception as e:
                logger.warning(f"Error processing sheet '{sheet_name}': {str(e)}")
                continue

        if not all_data:
            raise ValueError("No valid data found in any sheet")

        # Combine all sheets
        combined_df = pd.concat(all_data, ignore_index=True)

        # Clean data
        combined_df = combined_df.dropna()
        combined_df['Year'] = combined_df['Year'].astype(int)
        combined_df['Value'] = pd.to_numeric(combined_df['Value'], errors='coerce')
        combined_df = combined_df.dropna(subset=['Value'])

        # Pivot to wide format for correlation analysis
        df_wide = combined_df.pivot_table(
            index=['Unit', 'Year'],
            columns='Indicator',
            values='Value',
            aggfunc='first'  # In case of duplicates, take first value
        ).reset_index()

        # Flatten column names
        df_wide.columns.name = None

        # Data validation
        n_units = df_wide['Unit'].nunique()
        n_years = df_wide['Year'].nunique()
        n_indicators = len(df_wide.columns) - 2  # Subtract Unit and Year columns

        logger.info(f"Multi-sheet correlation data loaded successfully:")
        logger.info(f"  - Shape: {df_wide.shape}")
        logger.info(f"  - Units: {n_units}")
        logger.info(f"  - Years: {n_years} (range: {df_wide['Year'].min()} - {df_wide['Year'].max()})")
        logger.info(f"  - Indicators: {n_indicators}")
        logger.info(f"  - Data completeness: {df_wide.notna().mean().mean():.1%}")

        return df_wide

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading multi-sheet correlation data from {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


@recoverable
@safe_exception_handler
def load_correlation_data(file_path: str, unit_col: str = 'Unit', year_col: str = 'Year',
                          alt_unit_cols: List[str] = None, alt_year_cols: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Load data for correlation analysis from CSV, Excel or Parquet files.

    This function handles the new data format for correlation analysis where:
    - First column: Research units (companies, countries, etc.)
    - Second column: Year
    - Remaining columns: Indicators/variables

    Args:
        file_path (str): Path to CSV, Excel or Parquet file
        unit_col (str): Name of the unit column (default: 'Unit')
        year_col (str): Name of the year column (default: 'Year')

    Returns:
        Optional[pd.DataFrame]: DataFrame with the loaded data, or None if loading fails

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid
    """
    try:
        # Auto-detect file type and load accordingly
        file_path_str = str(file_path).lower()
        
        if file_path_str.endswith('.parquet'):
            return load_correlation_data_parquet(file_path, unit_col, year_col, alt_unit_cols, alt_year_cols)
        elif file_path_str.endswith(('.xlsx', '.xls')):
            # For Excel files, use existing correlation data loader
            all_sheets_data = load_excel_file(file_path)
            if not all_sheets_data:
                return None
            
            # Assuming the first sheet contains the correlation data
            first_sheet_name = list(all_sheets_data.keys())[0]
            df = all_sheets_data[first_sheet_name]
            
            # Rename columns to match expected format if needed
            if len(df.columns) >= 3 and df.columns[0] != unit_col:
                df = df.rename(columns={df.columns[0]: unit_col, df.columns[1]: year_col})
                
            return df
            
        elif file_path_str.endswith('.csv'):
            # Use existing CSV loader
            all_data = load_csv_file(file_path)
            if not all_data:
                return None
            
            # Return the 'data' DataFrame
            return all_data['data']
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or Parquet files.")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading correlation data from {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def validate_correlation_data(df: pd.DataFrame, unit_col: str = 'Unit', year_col: str = 'Year') -> Dict[str, Any]:
    """
    Validate correlation data and provide quality metrics.

    Args:
        df: DataFrame to validate
        unit_col: Name of unit column
        year_col: Name of year column

    Returns:
        Dictionary with validation results and metrics
    """
    try:
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        # Check required columns
        required_cols = [unit_col, year_col]
        for col in required_cols:
            if col not in df.columns:
                validation_results['errors'].append(f"Missing required column: {col}")
                validation_results['is_valid'] = False

        if not validation_results['is_valid']:
            return validation_results

        # Basic metrics
        n_units = df[unit_col].nunique()
        n_years = df[year_col].nunique()
        n_indicators = len(df.columns) - 2  # Subtract unit and year columns

        validation_results['metrics'] = {
            'n_units': n_units,
            'n_years': n_years,
            'n_indicators': n_indicators,
            'total_observations': len(df),
            'expected_observations': n_units * n_years
        }

        # Check data completeness
        indicator_cols = [col for col in df.columns if col not in required_cols]
        completeness_matrix = df[indicator_cols].notna()
        overall_completeness = completeness_matrix.mean().mean()

        validation_results['metrics']['overall_completeness'] = overall_completeness

        # Check for units with insufficient data
        unit_completeness = df.groupby(unit_col)[indicator_cols].apply(lambda x: x.notna().mean().mean())
        units_low_data = unit_completeness[unit_completeness < 0.3]  # Less than 30% complete

        if len(units_low_data) > 0:
            validation_results['warnings'].append(
                f"{len(units_low_data)} units have less than 30% data completeness"
            )

        # Check for time series length
        year_counts = df.groupby(unit_col)[year_col].count()
        min_years = year_counts.min()
        max_years = year_counts.max()

        validation_results['metrics'].update({
            'min_years_per_unit': min_years,
            'max_years_per_unit': max_years,
            'avg_years_per_unit': year_counts.mean()
        })

        if min_years < 3:
            validation_results['warnings'].append(
                "Some units have very short time series (< 3 years), which may affect correlation analysis"
            )

        # Success checks
        if n_units < 2:
            validation_results['errors'].append("Need at least 2 units for correlation analysis")
            validation_results['is_valid'] = False

        if n_indicators < 1:
            validation_results['errors'].append("Need at least 1 indicator for correlation analysis")
            validation_results['is_valid'] = False

        return validation_results

    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'metrics': {}
        }
