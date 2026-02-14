# pca_module.py
"""
Módulo de Análisis de Componentes Principales (PCA) para datos socioeconómicos.

Este módulo implementa funciones especializadas para:
- Realizar PCA con configuraciones flexibles
- Calcular y analizar varianza explicada
- Generar visualizaciones de scree plots
- Extraer cargas (loadings) de componentes principales
- Estandarizar direcciones de componentes para interpretación consistente

Las funciones están optimizadas para análisis de datos socioeconómicos donde
la interpretabilidad y robustez son fundamentales.

Autor: David Armando Abreu Rosique
Fecha: 2025
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import traceback
from typing import Tuple, Optional, Union, List

# Importar sistemas de optimización y configuración
from backend.logging_config import get_logger
from backend.performance_optimizer import cached, profiled, optimize_memory
from backend.config_manager import get_config
from backend.data_validation import validate_dataframe_for_pca, DataValidator
from backend.error_recovery import recoverable

logger = get_logger("pca_module")


def print_module_pca(mensaje: str) -> None:
    """Imprime mensajes con prefijo del módulo para debugging."""
    logger.debug(f"MODULE PCA: {mensaje}")


@recoverable
@profiled
@cached
def realizar_pca(
    df_estandarizado: pd.DataFrame,
    n_components: Optional[Union[int, float, str]] = None,
) -> Tuple[Optional[PCA], Optional[pd.DataFrame]]:
    """
    Realiza el Análisis de Componentes Principales (ACP) sobre un DataFrame estandarizado.

    Esta función implementa PCA usando scikit-learn con configuraciones flexibles
    para diferentes tipos de análisis. Maneja automáticamente casos edge como
    datasets pequeños o con características limitadas.

    Args:
        df_estandarizado (pd.DataFrame): DataFrame con datos estandarizados (media=0, std=1).
            Las filas deben representar observaciones (ej. años o países) y las columnas
            variables (ej. indicadores socioeconómicos). No debe contener valores NaN.
        n_components (Optional[Union[int, float, str]]): Número de componentes a retener.
            - int: número exacto de componentes (ej. 3)
            - float (0.0-1.0): porcentaje de varianza a explicar (ej. 0.95 para 95%)
            - str 'mle': usa estimador Maximum Likelihood de Minka
            - None: retiene min(n_muestras, n_features) componentes

    Returns:
        Tuple[Optional[PCA], Optional[pd.DataFrame]]:
            - pca_model: Objeto PCA ajustado de scikit-learn, None si hay error
            - df_pca_components: DataFrame con componentes principales como columnas
              (PC1, PC2, ...) y observaciones como filas, None si hay error

    Raises:
        ValueError: Si el DataFrame contiene NaN o no es numérico

    Example:
        >>> # PCA básico con todos los componentes
        >>> pca_model, df_components = realizar_pca(df_std)
        >>> print(f"Componentes generados: {df_components.shape[1]}")

        >>> # PCA para explicar 95% de varianza
        >>> pca_model, df_components = realizar_pca(df_std, n_components=0.95)

        >>> # PCA con número fijo de componentes
        >>> pca_model, df_components = realizar_pca(df_std, n_components=3)

    Note:
        - Se establece random_state=42 para reproducibilidad
        - La función valida automáticamente que n_components no exceda el máximo posible
        - Los nombres de componentes siguen el formato PC1, PC2, PC3, etc.
        - El índice del DataFrame resultante se preserva del DataFrame original
    """
    logger.info(
        f"Iniciando ACP con n_components={n_components if n_components is not None else 'todos'}."
    )

    # Usar configuración para valores por defecto
    config = get_config().pca

    # Manejar casos especiales de n_components
    if n_components is None:
        n_components_to_use = config.default_n_components
        # Si el default también es problemático, usar None (todos los componentes)
        if n_components_to_use == "auto":
            n_components_to_use = None
    elif n_components == "auto":
        # 'auto' no es válido en scikit-learn, convertir a None
        n_components_to_use = None
        logger.warning(
            "Parámetro 'auto' no es válido para n_components, usando None (todos los componentes)"
        )
    else:
        n_components_to_use = n_components

    # Validar datos de entrada usando la función independiente
    try:
        is_valid, validation_info = validate_dataframe_for_pca(
            df_estandarizado, 2  # Allow at least 2 samples for PCA
        )
        if not is_valid:
            errors = validation_info.get('errors', [])
            summary = validation_info.get('summary', 'Error desconocido')
            error_detail = '; '.join(errors) if errors else summary
            error_msg = f"Validación fallida: {error_detail}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError:
        raise  # Re-raise our own ValueError
    except Exception as e:
        logger.error(f"Error durante validación: {e}")
        raise ValueError(f"Error durante validación de datos: {e}") from e

    # Optimizar memoria del DataFrame
    df_estandarizado = optimize_memory(df_estandarizado)

    if df_estandarizado is None or df_estandarizado.empty:
        logger.error("El DataFrame estandarizado de entrada está vacío o es None.")
        raise ValueError("El DataFrame estandarizado está vacío. Verifica que los datos contengan valores válidos.")

    if df_estandarizado.isnull().sum().sum() > 0:
        nan_count = df_estandarizado.isnull().sum().sum()
        total = df_estandarizado.size
        pct = (nan_count / total) * 100
        logger.error(
            "El DataFrame estandarizado contiene valores NaN. El ACP no puede continuar."
        )
        logger.error(
            f"NaNs por columna:\n{df_estandarizado.isnull().sum()[df_estandarizado.isnull().sum() > 0]}"
        )
        raise ValueError(
            f"Los datos contienen {nan_count} valores faltantes ({pct:.1f}%). "
            f"El ACP requiere datos completos. Intenta con menos indicadores o más unidades."
        )

    try:
        # Asegurar que todas las columnas sean numéricas (aunque deberían serlo después de estandarizar)
        numeric_cols = df_estandarizado.select_dtypes(include=np.number).columns
        if len(numeric_cols) != df_estandarizado.shape[1]:
            logger.warning(
                "No todas las columnas en el DataFrame estandarizado son numéricas."
            )
            logger.warning("Se procederá solo con las columnas numéricas.")
            df_procesar_pca = df_estandarizado[numeric_cols]
            if df_procesar_pca.empty:
                logger.error(
                    "No hay columnas numéricas para el ACP después del filtrado."
                )
                return None, None
        else:
            df_procesar_pca = (
                df_estandarizado.copy()
            )  # Usar una copia para evitar SettingWithCopyWarning si se modifica

        # Si n_components es None, PCA() por defecto usa min(n_samples, n_features)
        # Si n_components es un float para varianza explicada, y el número de componentes resultante
        # es menor que 1, puede dar problemas. PCA maneja esto internamente.

        # Considerar el caso donde n_components solicitado podría ser mayor que min(n_samples, n_features)
        max_possible_components = min(
            df_procesar_pca.shape[0], df_procesar_pca.shape[1]
        )

        if (
            isinstance(n_components_to_use, int)
            and n_components_to_use > max_possible_components
        ):
            logger.warning(
                f"n_components ({n_components_to_use}) es mayor que el máximo posible ({max_possible_components})."
            )
            logger.warning(f"Se ajustará n_components a {max_possible_components}.")
            n_components_pca = max_possible_components
        elif n_components_to_use is None:  # Default
            n_components_pca = None  # PCA determinará automáticamente
        else:
            n_components_pca = n_components_to_use

        logger.info(f"Usando n_components={n_components_pca} para PCA")
        pca_model = PCA(
            n_components=n_components_pca, random_state=config.random_state
        )  # random_state para reproducibilidad

        # Ajustar el modelo y transformar los datos
        # .fit_transform() ajusta el modelo y luego aplica la reducción de dimensionalidad.
        pca_data = pca_model.fit_transform(df_procesar_pca)

        num_componentes_reales = pca_model.n_components_
        nombres_componentes = [f"PC{i+1}" for i in range(num_componentes_reales)]

        df_pca_components = pd.DataFrame(
            data=pca_data, columns=nombres_componentes, index=df_procesar_pca.index
        )

        logger.info(
            f"ACP realizado. Número de componentes generados: {num_componentes_reales}."
        )
        logger.info(
            f"Forma del DataFrame de componentes principales: {df_pca_components.shape}"
        )

        return pca_model, df_pca_components

    except Exception as e:
        logger.error(f"Error durante la ejecución del ACP: {e}")
        logger.error(traceback.format_exc())
        return None, None


def obtener_varianza_explicada(pca_model):
    """
    Obtiene la varianza explicada individual y acumulada de un modelo PCA ajustado.

    Args:
        pca_model (sklearn.decomposition.PCA): El objeto PCA ajustado.

    Returns:
        tuple: (explained_variance_ratio, cumulative_explained_variance)
            - explained_variance_ratio (np.array): Varianza explicada por cada componente.
                                                   None si el modelo no es válido.
            - cumulative_explained_variance (np.array): Varianza explicada acumulada.
                                                        None si el modelo no es válido.
    """
    print_module_pca("Obteniendo varianza explicada del modelo PCA.")

    if pca_model is None:
        print_module_pca("Error: El modelo PCA proporcionado es None.")
        return None, None

    try:
        # Atributo de scikit-learn PCA, almacena el porcentaje de varianza explicado por cada componente.
        explained_variance_ratio = pca_model.explained_variance_ratio_

        # Calculamos la varianza acumulada
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        print_module_pca(
            f"Varianza explicada por componente (ratio): {explained_variance_ratio}"
        )
        print_module_pca(
            f"Varianza explicada acumulada: {cumulative_explained_variance}"
        )

        return explained_variance_ratio, cumulative_explained_variance

    except AttributeError:
        print_module_pca(
            "Error: El objeto PCA no parece estar ajustado o no tiene el atributo 'explained_variance_ratio_'."
        )
        return None, None
    except Exception as e:
        print_module_pca(f"Error al obtener la varianza explicada: {e}")
        traceback.print_exc()
        return None, None


def graficar_scree_plot(explained_variance_ratio):
    """
    Genera y muestra un "scree plot" para visualizar la varianza explicada
    por cada componente principal y la varianza acumulada.

    Args:
        explained_variance_ratio (np.array): Un array con la varianza explicada
                                             por cada componente principal.
                                             (Salida de obtener_varianza_explicada).
    """
    print_module_pca("Generando Scree Plot.")

    if explained_variance_ratio is None or explained_variance_ratio.size == 0:
        print_module_pca("Error: No hay datos de varianza explicada para graficar.")
        return

    try:
        num_components = len(explained_variance_ratio)
        component_numbers = np.arange(1, num_components + 1)  # PC1, PC2, ...

        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(10, 6))  # Tamaño adecuado para la visualización

        # Graficar la varianza explicada individual (barras)
        plt.bar(
            component_numbers,
            explained_variance_ratio,
            alpha=0.7,
            align="center",
            label="Varianza explicada individual",
            color="dodgerblue",
        )

        # Graficar la varianza explicada acumulada (línea)
        plt.plot(
            component_numbers,
            cumulative_explained_variance,
            marker="o",
            linestyle="-",
            label="Varianza explicada acumulada",
            color="crimson",
        )

        # Líneas de referencia comunes (opcional, pero útil)
        plt.axhline(
            y=0.9,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="90% Varianza Acumulada",
        )
        plt.axhline(
            y=0.95,
            color="lightgray",
            linestyle="--",
            linewidth=0.8,
            label="95% Varianza Acumulada",
        )

        plt.title(
            "Scree Plot - Varianza Explicada por Componente Principal", fontsize=15
        )
        plt.xlabel("Componente Principal", fontsize=12)
        plt.ylabel("Proporción de Varianza Explicada", fontsize=12)

        plt.xticks(
            component_numbers
        )  # Asegurar que todos los números de componentes se muestren
        plt.yticks(
            np.arange(0, 1.1, 0.1)
        )  # Marcas en el eje Y de 0 a 1 con pasos de 0.1

        plt.legend(loc="best")
        plt.grid(axis="y", linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.show()

        print_module_pca("Scree Plot mostrado.")

    except Exception as e:
        print_module_pca(f"Error al generar el scree plot: {e}")
        traceback.print_exc()


def obtener_cargas_pca(pca_model, nombres_indicadores_originales):
    """
    Obtiene las cargas (loadings) de los componentes principales.
    Las cargas indican cómo cada variable original contribuye a cada componente principal.

    Args:
        pca_model (sklearn.decomposition.PCA): El objeto PCA ajustado.
        nombres_indicadores_originales (list): Lista de strings con los nombres
                                               de las variables originales (columnas del
                                               DataFrame que se usó para ajustar PCA).

    Returns:
        pd.DataFrame: Un DataFrame donde las filas son los indicadores originales
                      y las columnas son los componentes principales (PC1, PC2, ...),
                      mostrando las cargas. Retorna None si hay error.
    """
    print_module_pca("Obteniendo cargas de los componentes principales.")

    if pca_model is None:
        print_module_pca("Error: El modelo PCA proporcionado es None.")
        return None

    if not nombres_indicadores_originales:
        print_module_pca(
            "Error: La lista de nombres de indicadores originales está vacía."
        )
        return None

    try:
        # pca_model.components_ tiene forma (n_components, n_features)
        # Necesitamos transponerlo para que las features (indicadores) sean filas
        # y los componentes sean columnas.
        cargas_raw = pca_model.components_

        # Verificar que el número de features en el modelo PCA coincida con la lista de nombres
        if cargas_raw.shape[1] != len(nombres_indicadores_originales):
            print_module_pca(
                f"Error: Discrepancia en el número de features. "
                f"Modelo PCA tiene {cargas_raw.shape[1]} features, "
                f"pero se proporcionaron {len(nombres_indicadores_originales)} nombres."
            )
            return None

        num_componentes_reales = pca_model.n_components_
        nombres_componentes = [f"PC{i+1}" for i in range(num_componentes_reales)]

        df_cargas = pd.DataFrame(
            data=cargas_raw.T,  # Transponer
            columns=nombres_componentes,
            index=nombres_indicadores_originales,
        )

        print_module_pca(
            "Cargas de los componentes principales obtenidas exitosamente."
        )
        # print_module_pca(f"DataFrame de Cargas:\n{df_cargas.head()}") # Opcional: imprimir preview

        return df_cargas

    except AttributeError:
        print_module_pca(
            "Error: El objeto PCA no parece estar ajustado o no tiene el atributo 'components_'."
        )
        return None
    except Exception as e:
        print_module_pca(f"Error al obtener las cargas del PCA: {e}")
        traceback.print_exc()
        return None


def prompt_seleccionar_n_componentes(
    max_components, suggested_n_comp_90=None, suggested_n_comp_95=None
):
    """
    Solicita al usuario que ingrese el número de componentes principales a retener.

    Args:
        max_components (int): El número máximo de componentes disponibles.
        suggested_n_comp_90 (int, optional): Número de componentes sugerido para alcanzar ~90% de varianza.
        suggested_n_comp_95 (int, optional): Número de componentes sugerido para alcanzar ~95% de varianza.


    Returns:
        int: El número de componentes seleccionado por el usuario.
             Retorna max_components si la entrada es inválida después de varios intentos
             o si el usuario no ingresa nada y presiona Enter (asumiendo que se quieren todos).
             O podría retornar None para que main.py decida.
             Por ahora, si es inválido o vacío, default a max_components puede ser una opción.
    """
    print_module_pca("Selección del número de componentes a retener.")

    if max_components <= 0:
        print_module_pca(
            "Error: No hay componentes disponibles para seleccionar (max_components <= 0)."
        )
        return 0  # O manejar de otra forma, e.g., None o raise error

    while True:
        prompt_message = f"Ingresa el número de componentes principales que deseas retener (1-{max_components}).\n"
        if suggested_n_comp_90:
            prompt_message += f"  Sugerencia: {suggested_n_comp_90} componentes explican aprox. el 80% de la varianza.\n"
        if suggested_n_comp_95:
            prompt_message += f"  Sugerencia: {suggested_n_comp_95} componentes explican aprox. el 90% de la varianza.\n"
        prompt_message += f"Presiona Enter para usar todos ({max_components}): "

        try:
            user_input_str = input(prompt_message).strip()

            if not user_input_str:  # Usuario presionó Enter sin ingresar nada
                print_module_pca(
                    f"No se ingresó un número. Se usarán todos los {max_components} componentes."
                )
                return max_components

            n_seleccionados = int(user_input_str)

            if 1 <= n_seleccionados <= max_components:
                print_module_pca(f"Se seleccionaron {n_seleccionados} componentes.")
                return n_seleccionados
            else:
                print_module_pca(
                    f"Error: El número debe estar entre 1 y {max_components}. Intenta de nuevo."
                )

        except ValueError:
            print_module_pca(
                "Error: Entrada inválida. Por favor, ingresa un número entero. Intenta de nuevo."
            )
        except Exception as e:
            print_module_pca(
                f"Se produjo un error inesperado durante la selección: {e}"
            )
            print_module_pca(
                f"Se usarán todos los {max_components} componentes como fallback."
            )
            return max_components  # Fallback en caso de error no previsto


def estandarizar_direccion_pca(
    pca_model,
    df_pc_scores,
    df_estandarizado,
    anchor_variable_code,
    component_to_check=0,
):
    """
    Verifica la dirección de un componente principal y lo invierte si es necesario para mantener la consistencia.
    Se asegura de que una 'variable ancla' siempre tenga una carga positiva en el componente especificado.

    Args:
        pca_model: El objeto PCA ajustado de scikit-learn.
        df_pc_scores (pd.DataFrame): DataFrame con los scores de los componentes.
        df_estandarizado (pd.DataFrame): DataFrame que se usó para ajustar el PCA.
        anchor_variable_code (str): El nombre/código de la columna de la variable ancla.
        component_to_check (int): El índice del componente a verificar (0 para PC1, 1 para PC2, etc.).

    Returns:
        tuple: (pca_model, df_pc_scores) con la dirección potencialmente invertida.
    """
    try:
        # Encontrar el índice de la columna de nuestra variable ancla
        lista_columnas = df_estandarizado.columns.tolist()
        anchor_variable_index = lista_columnas.index(anchor_variable_code)

        # Obtener la carga de la variable ancla en el componente a revisar
        # pca_model.components_ tiene forma (n_components, n_features)
        loading_value = pca_model.components_[component_to_check, anchor_variable_index]

        # Si el signo de la carga es negativo, la dirección está "al revés"
        if loading_value < 0:
            logger.info(
                f"Invirtiendo la dirección de PC{component_to_check + 1} para consistencia (basado en '{anchor_variable_code}')."
            )
            # Invertir el signo de las cargas para ese componente
            pca_model.components_[component_to_check, :] *= -1
            # Invertir el signo de los scores para ese componente
            pc_name = f"PC{component_to_check + 1}"
            if pc_name in df_pc_scores.columns:
                df_pc_scores[pc_name] *= -1

    except ValueError:
        logger.warning(
            f"La variable ancla '{anchor_variable_code}' no se encontró en las columnas. No se pudo estandarizar la dirección del PCA."
        )
    except Exception as e:
        logger.error(f"Error al estandarizar la dirección del PCA: {e}")

    return pca_model, df_pc_scores
