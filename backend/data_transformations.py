"""
Data Transformations Module for PCA Analysis

Provides intelligent data transformations to handle skewed distributions,
especially common in financial and economic data (revenues, assets, market values).

This module addresses the scale problem in biplots by transforming variables
BEFORE standardization and PCA, improving both statistical validity and visualization.

Key Features:
- Automatic skewness detection
- Multiple transformation methods (log, Box-Cox, Yeo-Johnson)
- Smart column type detection (magnitude vs ratios)
- Preservation of original data for inverse transformations
- Detailed logging and diagnostics

Author: David Armando Abreu Rosique
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.stats import skew, boxcox
import warnings

# Import logging
try:
    from logging_config import get_logger
    logger = get_logger("data_transformations")
except ImportError:
    import logging
    logger = logging.getLogger("data_transformations")


class DataTransformer:
    """
    Intelligent data transformer for PCA preprocessing.
    
    Handles skewed distributions common in financial data by applying
    appropriate transformations before standardization.
    """
    
    def __init__(
        self,
        skewness_threshold: float = 1.0,
        auto_detect_magnitude_cols: bool = True,
        preserve_ratios: bool = True
    ):
        """
        Initialize DataTransformer.
        
        Args:
            skewness_threshold: Absolute skewness value above which to apply transformation
            auto_detect_magnitude_cols: Automatically detect magnitude columns
            preserve_ratios: Don't transform ratio/percentage columns
        """
        self.skewness_threshold = skewness_threshold
        self.auto_detect_magnitude_cols = auto_detect_magnitude_cols
        self.preserve_ratios = preserve_ratios
        
        # Storage for transformation metadata
        self.transformation_info: Dict[str, Dict[str, Any]] = {}
        self.original_columns: List[str] = []
        
    def detect_column_type(self, series: pd.Series, col_name: str) -> str:
        """
        Detect if a column is magnitude, ratio, or other type.
        
        Args:
            series: Data series to analyze
            col_name: Column name
            
        Returns:
            'magnitude', 'ratio', 'percentage', or 'other'
        """
        # Keywords indicating magnitude columns
        magnitude_keywords = [
            'ingreso', 'activo', 'pasivo', 'venta', 'revenue', 'asset', 'liability',
            'capital', 'utilidad', 'profit', 'valor', 'value', 'mercado', 'market',
            'empleado', 'employee', 'millon', 'million', 'size', 'tamaño', 'cantidad',
            'amount', 'total', 'patrimonio', 'equity', 'deuda', 'debt', 'costo', 'cost',
            'gasto', 'expense', 'credito', 'credit', 'precio', 'price', 'volumen', 'volume'
        ]
        
        # Keywords indicating ratios/percentages
        ratio_keywords = [
            'ratio', 'margin', 'margen', 'roe', 'roa', 'roi', 'porcentaj', 'percent',
            'tasa', 'rate', 'indice', 'index', 'coeficiente', 'coefficient', 'razon',
            'proporcion', 'proportion', 'eficiencia', 'efficiency', 'productividad',
            'productivity', 'rentabilidad', 'profitability', 'liquidez', 'liquidity',
            'apalancamiento', 'leverage', 'rotacion', 'turnover', 'crecimiento', 'growth'
        ]
        
        col_lower = col_name.lower()
        
        # Check for percentage indicators
        if any(kw in col_lower for kw in ['porcentaj', 'percent', '%']):
            return 'percentage'
        
        # Check for ratio indicators
        if any(kw in col_lower for kw in ratio_keywords):
            return 'ratio'
        
        # Check for magnitude indicators
        if any(kw in col_lower for kw in magnitude_keywords):
            # Additional check: magnitudes usually have large ranges
            if series.notna().any():
                value_range = series.max() - series.min()
                if value_range > 100:  # Likely a magnitude
                    return 'magnitude'
        
        # Check data range characteristics
        if series.notna().any():
            min_val = series.min()
            max_val = series.max()
            
            # Percentages typically 0-100 or 0-1
            if min_val >= 0 and max_val <= 1:
                return 'percentage'
            if min_val >= 0 and max_val <= 100 and 'tasa' not in col_lower:
                return 'percentage'
        
        return 'other'
    
    def calculate_skewness(self, series: pd.Series) -> float:
        """
        Calculate skewness of a series, handling edge cases.
        
        Args:
            series: Data series
            
        Returns:
            Skewness value (0.0 if cannot be calculated)
        """
        try:
            clean_series = series.dropna()
            if len(clean_series) < 3:
                return 0.0
            if clean_series.std() == 0:
                return 0.0
            return skew(clean_series)
        except Exception as e:
            logger.warning(f"Could not calculate skewness: {e}")
            return 0.0
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze DataFrame to determine which columns need transformation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with analysis results for each column
        """
        analysis = {}
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_type = self.detect_column_type(df[col], col)
            skewness = self.calculate_skewness(df[col])
            
            needs_transform = (
                abs(skewness) > self.skewness_threshold and
                col_type == 'magnitude'
            )
            
            analysis[col] = {
                'type': col_type,
                'skewness': skewness,
                'needs_transform': needs_transform,
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'has_negatives': (df[col] < 0).any(),
                'has_zeros': (df[col] == 0).any(),
                'recommended_method': self._recommend_method(df[col], col_type, skewness)
            }
        
        return analysis
    
    def _recommend_method(
        self,
        series: pd.Series,
        col_type: str,
        skewness: float
    ) -> Optional[str]:
        """
        Recommend transformation method based on data characteristics.
        
        Args:
            series: Data series
            col_type: Column type
            skewness: Skewness value
            
        Returns:
            Recommended method or None
        """
        if col_type != 'magnitude' or abs(skewness) <= self.skewness_threshold:
            return None
        
        has_negatives = (series < 0).any()
        has_zeros = (series == 0).any()
        
        if has_negatives:
            return 'yeo-johnson'  # Handles negative values
        elif has_zeros:
            return 'log1p'  # log(1+x) handles zeros
        else:
            return 'log'  # Standard log for positive values
    
    def transform(
        self,
        df: pd.DataFrame,
        method: str = 'auto',
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply transformations to DataFrame.
        
        Args:
            df: Input DataFrame
            method: Transformation method:
                - 'auto': Automatically detect and apply best method
                - 'log': np.log (requires positive values)
                - 'log1p': np.log1p (handles zeros)
                - 'sqrt': Square root
                - 'box-cox': Box-Cox transformation
                - 'yeo-johnson': Yeo-Johnson transformation
                - 'none': No transformation
            columns: Specific columns to transform (None = auto-detect)
            
        Returns:
            Tuple of (Transformed DataFrame, transformation_info)
        """
        df_transformed = df.copy()
        self.original_columns = df.columns.tolist()
        
        # Reset transformation info for this call
        self.transformation_info = {}
        self.auto_selected_methods = {}
        
        if method == 'none':
            logger.info("🔹 No transformation applied (method='none')")
            return df_transformed, {'auto_methods_used': {}, 'total_transformed': 0}
        
        # Analyze data
        analysis = self.analyze_data(df)
        
        # Determine columns to transform
        if columns is None and method == 'auto':
            columns_to_transform = [
                col for col, info in analysis.items()
                if info['needs_transform']
            ]
        elif columns is None:
            # Apply to all magnitude columns if manual method specified
            columns_to_transform = [
                col for col, info in analysis.items()
                if info['type'] == 'magnitude'
            ]
        else:
            columns_to_transform = columns
        
        if not columns_to_transform:
            logger.info("✅ No columns need transformation")
            return df_transformed, {'auto_methods_used': {}, 'total_transformed': 0}
        
        logger.info(f"🔄 Transforming {len(columns_to_transform)} columns...")
        
        # Apply transformations
        for col in columns_to_transform:
            if col not in df.columns:
                continue
            
            col_info = analysis.get(col, {})
            transform_method = method if method != 'auto' else col_info.get('recommended_method', 'log1p')
            
            if transform_method is None:
                continue
            
            # Track auto-selection
            if method == 'auto':
                self.auto_selected_methods[col] = transform_method
                logger.info(f"🎯 Auto-selected method for {col}: {transform_method}")
            
            try:
                original_skewness = col_info.get('skewness', 0)
                df_transformed[col] = self._apply_single_transform(
                    df[col],
                    transform_method
                )
                new_skewness = self.calculate_skewness(df_transformed[col])
                
                # Store transformation info
                self.transformation_info[col] = {
                    'method': transform_method,
                    'original_skewness': original_skewness,
                    'new_skewness': new_skewness,
                    'type': col_info.get('type', 'unknown'),
                    'auto_selected': method == 'auto'
                }
                
                logger.info(
                    f"  ✅ {col}: {transform_method} | "
                    f"skewness {original_skewness:.2f} → {new_skewness:.2f}"
                )
                
            except Exception as e:
                logger.warning(f"  ⚠️  Could not transform {col}: {e}")
                self.transformation_info[col] = {
                    'method': 'failed',
                    'error': str(e),
                    'auto_selected': method == 'auto'
                }
        
        # Return transformation info
        transformation_info = {
            'auto_methods_used': self.auto_selected_methods,
            'total_transformed': len(self.transformation_info),
            'requested_method': method,
            'actual_methods_applied': {col: info['method'] for col, info in self.transformation_info.items() if info['method'] != 'failed'}
        }
        
        return df_transformed, transformation_info
    
    def _apply_single_transform(
        self,
        series: pd.Series,
        method: str
    ) -> pd.Series:
        """
        Apply a single transformation method to a series.
        
        Args:
            series: Input series
            method: Transformation method
            
        Returns:
            Transformed series
        """
        clean_series = series.dropna()
        
        if method == 'log':
            # Ensure positive values
            if (clean_series <= 0).any():
                raise ValueError("log requires positive values, use log1p instead")
            transformed = np.log(series)
            
        elif method == 'log1p':
            # Handle zeros and negative values by ensuring non-negative
            if (clean_series < 0).any():
                # Shift to non-negative
                min_val = clean_series.min()
                transformed = np.log1p(series - min_val)
            else:
                transformed = np.log1p(series)
        
        elif method == 'sqrt':
            # Requires non-negative values
            if (clean_series < 0).any():
                raise ValueError("sqrt requires non-negative values")
            transformed = np.sqrt(series)
        
        elif method == 'box-cox':
            # Requires positive values
            if (clean_series <= 0).any():
                raise ValueError("box-cox requires positive values")
            from scipy.stats import boxcox
            # Handle NaNs: only transform non-NaN values
            mask = series.notna()
            transformed = series.copy()
            if mask.sum() > 1:
                transformed_values, fitted_lambda = boxcox(series[mask].values)
                transformed[mask] = transformed_values
        
        elif method == 'yeo-johnson':
            # Works with any values (negative, zero, positive)
            from sklearn.preprocessing import PowerTransformer
            transformer = PowerTransformer(method='yeo-johnson')
            # Handle NaNs
            mask = series.notna()
            transformed = series.copy()
            if mask.sum() > 0:
                values_to_transform = series[mask].values.reshape(-1, 1)
                transformed_values = transformer.fit_transform(values_to_transform)
                transformed[mask] = transformed_values.flatten()
        
        else:
            raise ValueError(f"Unknown transformation method: {method}")
        
        return transformed
    
    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get summary of applied transformations.
        
        Returns:
            DataFrame with transformation details
        """
        if not self.transformation_info:
            return pd.DataFrame()
        
        summary_data = []
        for col, info in self.transformation_info.items():
            summary_data.append({
                'Column': col,
                'Method': info.get('method', 'unknown'),
                'Type': info.get('type', 'unknown'),
                'Original_Skewness': info.get('original_skewness', np.nan),
                'New_Skewness': info.get('new_skewness', np.nan),
                'Improvement': abs(info.get('original_skewness', 0)) - abs(info.get('new_skewness', 0))
            })
        
        return pd.DataFrame(summary_data)


# Convenience functions for quick usage

def transform_financial_data(
    df: pd.DataFrame,
    skewness_threshold: float = 1.0,
    method: str = 'auto'
) -> Tuple[pd.DataFrame, DataTransformer, Dict[str, Any]]:
    """
    Quick transformation for financial/economic data.
    
    Args:
        df: Input DataFrame
        skewness_threshold: Threshold for transformation
        method: Transformation method
        
    Returns:
        Tuple of (transformed DataFrame, transformer object, transformation_info)
    """
    transformer = DataTransformer(skewness_threshold=skewness_threshold)
    df_transformed, transformation_info = transformer.transform(df, method=method)
    
    return df_transformed, transformer, transformation_info


def analyze_data_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze data distribution and provide recommendations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Analysis DataFrame
    """
    transformer = DataTransformer()
    analysis = transformer.analyze_data(df)
    
    analysis_data = []
    for col, info in analysis.items():
        analysis_data.append({
            'Column': col,
            'Type': info['type'],
            'Skewness': info['skewness'],
            'Needs_Transform': info['needs_transform'],
            'Recommended_Method': info.get('recommended_method', 'none'),
            'Min': info['min_value'],
            'Max': info['max_value'],
            'Has_Negatives': info['has_negatives'],
            'Has_Zeros': info['has_zeros']
        })
    
    return pd.DataFrame(analysis_data)


if __name__ == "__main__":
    # Test the transformer
    print("🧪 Testing Data Transformations Module\n")
    
    # Create sample financial data with skew
    np.random.seed(42)
    n_companies = 50
    
    data = {
        'Ingresos_Millones': np.abs(np.random.lognormal(5, 2, n_companies)),
        'Activos_Millones': np.abs(np.random.lognormal(6, 1.5, n_companies)),
        'Empleados': np.abs(np.random.lognormal(4, 1, n_companies)),
        'ROE_Porcentaje': np.random.normal(15, 5, n_companies),
        'Margen_Utilidad': np.random.normal(0.1, 0.05, n_companies),
        'Ratio_Deuda': np.random.normal(0.5, 0.2, n_companies)
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Original Data Summary:")
    print(df.describe())
    
    print("\n🔍 Distribution Analysis:")
    analysis = analyze_data_distribution(df)
    print(analysis.to_string(index=False))
    
    print("\n🔄 Applying Transformations...")
    df_transformed, transformer = transform_financial_data(df)
    
    print("\n📈 Transformation Summary:")
    summary = transformer.get_transformation_summary()
    print(summary.to_string(index=False))
    
    print("\n📊 Transformed Data Summary:")
    print(df_transformed.describe())
    
    print("\n✅ Test completed!")
