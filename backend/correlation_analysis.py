# correlation_analysis.py
"""
Correlation Analysis Module for PCA Application

This module provides advanced correlation and similarity analysis between research units,
including traditional statistical correlations and time series methods like Dynamic Time Warping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

# Optional imports for advanced functionality
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    warnings.warn("dtaidistance not available. DTW correlation will not be supported.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Network analysis features will be limited.")

# Import logging
try:
    from logging_config import get_logger
    logger = get_logger("correlation_analysis")
except ImportError:
    import logging
    logger = logging.getLogger("correlation_analysis")


class CorrelationAnalyzer:
    """Advanced correlation and similarity analysis for research units."""

    def __init__(self):
        self.logger = logger

    def calculate_similarity_matrix(self, df: pd.DataFrame,
                                  method: str = 'pearson',
                                  time_aggregated: bool = True,
                                  min_periods: int = 3) -> pd.DataFrame:
        """
        Calculate similarity/correlation matrix between research units.

        Args:
            df: DataFrame with columns [Unit, Year, Indicator1, Indicator2, ...]
            method: Correlation method ('pearson', 'spearman', 'kendall', 'dtw')
            time_aggregated: If True, aggregate data by unit before correlation
            min_periods: Minimum periods required for correlation

        Returns:
            DataFrame: Similarity matrix with units as both rows and columns
        """
        try:
            self.logger.info(f"Calculating similarity matrix using method: {method}")

            if df.empty:
                raise ValueError("Input DataFrame is empty")

            # Validate required columns
            required_cols = ['Unit', 'Year']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get indicator columns (all columns except Unit and Year)
            indicator_cols = [col for col in df.columns if col not in ['Unit', 'Year']]
            if not indicator_cols:
                raise ValueError("No indicator columns found")

            if time_aggregated:
                # Aggregate data by unit (mean across years)
                unit_data = df.groupby('Unit')[indicator_cols].mean().dropna()
                if unit_data.empty:
                    raise ValueError("No valid data after aggregation")

                return self._calculate_aggregated_correlation(unit_data, method, min_periods)
            else:
                # Time series correlation
                return self._calculate_time_series_correlation(df, indicator_cols, method, min_periods)

        except Exception as e:
            error_msg = f"Error calculating similarity matrix: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _calculate_aggregated_correlation(self, unit_data: pd.DataFrame,
                                        method: str, min_periods: int) -> pd.DataFrame:
        """Calculate correlation on aggregated unit data."""
        try:
            if method == 'dtw':
                # For aggregated data, DTW doesn't make sense, fall back to Pearson
                self.logger.warning("DTW not applicable for aggregated data, using Pearson correlation")
                method = 'pearson'

            if method in ['pearson', 'spearman', 'kendall']:
                corr_matrix = unit_data.T.corr(method=method, min_periods=min_periods)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")

            # Fill NaN values with 0 (no correlation)
            corr_matrix = corr_matrix.fillna(0)

            return corr_matrix

        except Exception as e:
            raise ValueError(f"Error in aggregated correlation calculation: {str(e)}")

    def _calculate_time_series_correlation(self, df: pd.DataFrame,
                                         indicator_cols: List[str],
                                         method: str, min_periods: int) -> pd.DataFrame:
        """Calculate correlation considering time series patterns."""
        try:
            units = df['Unit'].unique()
            n_units = len(units)

            if n_units < 2:
                raise ValueError("Need at least 2 units for correlation analysis")

            # Initialize similarity matrix
            similarity_matrix = pd.DataFrame(
                np.zeros((n_units, n_units)),
                index=units,
                columns=units
            )

            if method == 'dtw':
                if not DTW_AVAILABLE:
                    raise ValueError("DTW method requires dtaidistance library")

                return self._calculate_dtw_similarity(df, indicator_cols, units, similarity_matrix)
            else:
                # Traditional time series correlation
                return self._calculate_traditional_time_correlation(
                    df, indicator_cols, units, method, min_periods, similarity_matrix
                )

        except Exception as e:
            raise ValueError(f"Error in time series correlation calculation: {str(e)}")

    def _calculate_dtw_similarity(self, df: pd.DataFrame, indicator_cols: List[str],
                                units: List[str], similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate Dynamic Time Warping similarity between time series."""
        try:
            self.logger.info("Calculating DTW similarity for time series")

            # For each pair of units, calculate DTW distance for each indicator
            for i, unit1 in enumerate(units):
                for j, unit2 in enumerate(units):
                    if i == j:
                        similarity_matrix.loc[unit1, unit2] = 1.0  # Self-similarity
                        continue

                    if i > j:  # Matrix is symmetric
                        continue

                    # Get time series for both units
                    ts1 = df[df['Unit'] == unit1].sort_values('Year')[indicator_cols]
                    ts2 = df[df['Unit'] == unit2].sort_values('Year')[indicator_cols]

                    # Handle missing data by forward/backward fill
                    ts1 = ts1.fillna(method='ffill').fillna(method='bfill')
                    ts2 = ts2.fillna(method='ffill').fillna(method='bfill')

                    if ts1.empty or ts2.empty or len(ts1) < 2 or len(ts2) < 2:
                        similarity = 0.0
                    else:
                        # Calculate DTW distance for each indicator and average
                        distances = []
                        for col in indicator_cols:
                            if col in ts1.columns and col in ts2.columns:
                                s1 = ts1[col].values
                                s2 = ts2[col].values

                                # Normalize series
                                s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
                                s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)

                                try:
                                    distance = dtw.distance(s1_norm, s2_norm)
                                    # Convert distance to similarity (higher values = more similar)
                                    similarity_score = 1 / (1 + distance)
                                    distances.append(similarity_score)
                                except Exception as e:
                                    self.logger.warning(f"DTW calculation failed for {unit1}-{unit2}, {col}: {e}")
                                    continue

                        similarity = np.mean(distances) if distances else 0.0

                    similarity_matrix.loc[unit1, unit2] = similarity
                    similarity_matrix.loc[unit2, unit1] = similarity  # Symmetric

            return similarity_matrix

        except Exception as e:
            raise ValueError(f"Error in DTW similarity calculation: {str(e)}")

    def _calculate_traditional_time_correlation(self, df: pd.DataFrame, indicator_cols: List[str],
                                              units: List[str], method: str, min_periods: int,
                                              similarity_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional correlation on time-aligned series."""
        try:
            self.logger.info(f"Calculating {method} correlation for time series")

            # Pivot data to have units as columns, years as index
            correlation_data = {}

            for indicator in indicator_cols:
                pivot_df = df.pivot(index='Year', columns='Unit', values=indicator)

                # Calculate correlation matrix for this indicator
                if method in ['pearson', 'spearman', 'kendall']:
                    corr_matrix = pivot_df.corr(method=method, min_periods=min_periods)
                    correlation_data[indicator] = corr_matrix

            if not correlation_data:
                raise ValueError("No valid correlation data calculated")

            # Average correlations across indicators
            avg_correlation = pd.concat(correlation_data.values()).groupby(level=0).mean()

            # Fill the similarity matrix
            for unit1 in units:
                for unit2 in units:
                    if unit1 in avg_correlation.index and unit2 in avg_correlation.columns:
                        similarity_matrix.loc[unit1, unit2] = avg_correlation.loc[unit1, unit2]
                    else:
                        similarity_matrix.loc[unit1, unit2] = 0.0

            # Fill NaN values
            similarity_matrix = similarity_matrix.fillna(0)

            return similarity_matrix

        except Exception as e:
            raise ValueError(f"Error in traditional time correlation calculation: {str(e)}")

    def filter_similarity_matrix(self, similarity_matrix: pd.DataFrame,
                               threshold: float = 0.0,
                               min_connections: int = 1) -> pd.DataFrame:
        """
        Filter similarity matrix based on threshold and minimum connections.

        Args:
            similarity_matrix: Input similarity matrix
            threshold: Minimum similarity value to keep
            min_connections: Minimum number of connections per unit

        Returns:
            Filtered similarity matrix
        """
        try:
            # Apply threshold
            filtered_matrix = similarity_matrix.copy()
            filtered_matrix[filtered_matrix.abs() < threshold] = 0

            # Ensure minimum connections
            if min_connections > 1:
                for unit in filtered_matrix.index:
                    connections = (filtered_matrix.loc[unit].abs() >= threshold).sum() - 1  # Exclude self
                    if connections < min_connections:
                        # Keep only the strongest connections
                        row_data = filtered_matrix.loc[unit].copy()
                        row_data[unit] = 0  # Exclude self
                        threshold_value = np.sort(row_data.abs().values)[-min_connections] if len(row_data) > min_connections else 0
                        filtered_matrix.loc[unit] = np.where(row_data.abs() >= threshold_value, row_data, 0)

            return filtered_matrix

        except Exception as e:
            self.logger.error(f"Error filtering similarity matrix: {e}")
            return similarity_matrix

    def get_similarity_statistics(self, similarity_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the similarity matrix."""
        try:
            # Remove diagonal (self-similarities)
            off_diagonal = similarity_matrix.values
            np.fill_diagonal(off_diagonal, np.nan)

            stats_dict = {
                'mean_similarity': np.nanmean(off_diagonal),
                'std_similarity': np.nanstd(off_diagonal),
                'min_similarity': np.nanmin(off_diagonal),
                'max_similarity': np.nanmax(off_diagonal),
                'median_similarity': np.nanmedian(off_diagonal),
                'n_units': len(similarity_matrix),
                'n_connections': np.sum(~np.isnan(off_diagonal))
            }

            return stats_dict

        except Exception as e:
            self.logger.error(f"Error calculating similarity statistics: {e}")
            return {}


# Convenience functions
def calculate_similarity_matrix(df: pd.DataFrame, method: str = 'pearson',
                              time_aggregated: bool = True, min_periods: int = 3) -> pd.DataFrame:
    """Convenience function for similarity matrix calculation."""
    analyzer = CorrelationAnalyzer()
    return analyzer.calculate_similarity_matrix(df, method, time_aggregated, min_periods)


def filter_similarity_matrix(similarity_matrix: pd.DataFrame, threshold: float = 0.0,
                           min_connections: int = 1) -> pd.DataFrame:
    """Convenience function for similarity matrix filtering."""
    analyzer = CorrelationAnalyzer()
    return analyzer.filter_similarity_matrix(similarity_matrix, threshold, min_connections)


def get_similarity_statistics(similarity_matrix: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for similarity statistics."""
    analyzer = CorrelationAnalyzer()
    return analyzer.get_similarity_statistics(similarity_matrix)


# Test function
if __name__ == "__main__":
    # Example usage
    print("Testing correlation analysis module...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=10, freq='YS')
    units = ['Company_A', 'Company_B', 'Company_C', 'Company_D']

    data = []
    for unit in units:
        for i, date in enumerate(dates):
            # Generate correlated time series
            base_value = np.random.normal(100, 10)
            trend = i * 2
            noise = np.random.normal(0, 5)
            indicator1 = base_value + trend + noise
            indicator2 = base_value * 0.8 + trend * 0.5 + noise * 0.3

            data.append({
                'Unit': unit,
                'Year': date.year,
                'Indicator1': indicator1,
                'Indicator2': indicator2
            })

    df_sample = pd.DataFrame(data)
    print("Sample data shape:", df_sample.shape)
    print("Sample data preview:")
    print(df_sample.head())

    # Test correlation calculation
    analyzer = CorrelationAnalyzer()

    try:
        similarity_matrix = analyzer.calculate_similarity_matrix(df_sample, method='pearson')
        print("\nSimilarity matrix (Pearson):")
        print(similarity_matrix.round(3))

        stats = analyzer.get_similarity_statistics(similarity_matrix)
        print("\nSimilarity statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.3f}")

    except Exception as e:
        print(f"Error in testing: {e}")

    print("Correlation analysis module test completed.")