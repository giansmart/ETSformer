"""
Utilidades para cargar y procesar resultados de experimentos ETSformer.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class ResultsLoader:
    """Carga y procesa resultados de experimentos ETSformer."""

    def __init__(self, results_dir: str = "../results"):
        """
        Args:
            results_dir: Ruta al directorio de resultados
        """
        self.results_dir = Path(results_dir)

    def get_experiment_folders(self) -> List[Path]:
        """Obtiene todas las carpetas de experimentos."""
        folders = [f for f in self.results_dir.iterdir()
                  if f.is_dir() and not f.name.startswith('.')]
        return sorted(folders)

    def parse_folder_name(self, folder_name: str) -> Dict[str, str]:
        """
        Extrae información del nombre de la carpeta.

        Formato esperado: YYYYMMDD_HHMMSS_Model_Dataset_Horizon
        Ejemplo: 20250929_234430_ETSformer_ETTh1_24
        """
        parts = folder_name.split('_')
        if len(parts) >= 5:
            return {
                'date': parts[0],
                'time': parts[1],
                'model': parts[2],
                'dataset': parts[3],
                'horizon': int(parts[4])
            }
        return {}

    def load_test_results(self, folder: Path) -> Dict:
        """Carga resultados de test desde un experimento."""
        test_file = folder / "test_results_test.json"
        val_file = folder / "test_results_val.json"

        results = {}

        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
                results['test'] = test_data.get('metrics', {})

        if val_file.exists():
            with open(val_file, 'r') as f:
                val_data = json.load(f)
                results['val'] = val_data.get('metrics', {})

        return results

    def load_training_log(self, folder: Path) -> Dict:
        """Carga el log de entrenamiento."""
        log_file = folder / "training_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {}

    def get_all_results(self) -> pd.DataFrame:
        """
        Carga todos los resultados en un DataFrame.

        Returns:
            DataFrame con columnas: dataset, horizon, mae_test, mse_test,
                                   mae_val, mse_val, etc.
        """
        all_results = []

        for folder in self.get_experiment_folders():
            # Parsear nombre de carpeta
            info = self.parse_folder_name(folder.name)
            if not info:
                continue

            # Cargar métricas
            metrics = self.load_test_results(folder)

            # Crear fila de resultados
            row = {
                'dataset': info['dataset'],
                'horizon': info['horizon'],
                'model': info['model'],
                'date': info['date'],
                'time': info['time'],
                'folder': folder.name
            }

            # Agregar métricas de test
            if 'test' in metrics:
                row['mse_test'] = metrics['test'].get('mse')
                row['mae_test'] = metrics['test'].get('mae')
                row['rmse_test'] = metrics['test'].get('rmse')
                row['mape_test'] = metrics['test'].get('mape')
                row['mspe_test'] = metrics['test'].get('mspe')

            # Agregar métricas de validación
            if 'val' in metrics:
                row['mse_val'] = metrics['val'].get('mse')
                row['mae_val'] = metrics['val'].get('mae')
                row['rmse_val'] = metrics['val'].get('rmse')
                row['mape_val'] = metrics['val'].get('mape')
                row['mspe_val'] = metrics['val'].get('mspe')

            all_results.append(row)

        df = pd.DataFrame(all_results)

        # Ordenar por dataset y horizonte
        if not df.empty:
            df = df.sort_values(['dataset', 'horizon']).reset_index(drop=True)

        return df

    def get_training_history(self) -> pd.DataFrame:
        """
        Obtiene el historial de entrenamiento de todos los experimentos.

        Returns:
            DataFrame con columnas: dataset, horizon, epoch, train_loss,
                                   val_loss, test_loss, learning_rate
        """
        all_history = []

        for folder in self.get_experiment_folders():
            info = self.parse_folder_name(folder.name)
            if not info:
                continue

            log = self.load_training_log(folder)
            epochs = log.get('epochs', [])

            for epoch_data in epochs:
                row = {
                    'dataset': info['dataset'],
                    'horizon': info['horizon'],
                    'epoch': epoch_data.get('epoch'),
                    'train_loss': epoch_data.get('train_loss'),
                    'val_loss': epoch_data.get('val_loss'),
                    'test_loss': epoch_data.get('test_loss'),
                    'learning_rate': epoch_data.get('learning_rate'),
                    'epoch_time': epoch_data.get('epoch_time')
                }
                all_history.append(row)

        df = pd.DataFrame(all_history)
        return df

    def create_metrics_table(self, df: pd.DataFrame,
                            metrics: List[str] = ['mse_test', 'mae_test']) -> pd.DataFrame:
        """
        Crea una tabla pivote de métricas por dataset y horizonte.

        Args:
            df: DataFrame de resultados
            metrics: Lista de métricas a incluir

        Returns:
            DataFrame pivotado con datasets como filas y horizontes como columnas
        """
        tables = {}

        for metric in metrics:
            if metric in df.columns:
                pivot = df.pivot(index='dataset', columns='horizon', values=metric)
                tables[metric] = pivot

        return tables


def format_metric_table(table: pd.DataFrame, metric_name: str,
                       decimal_places: int = 4) -> pd.DataFrame:
    """
    Formatea una tabla de métricas para presentación.

    Args:
        table: Tabla pivote de métricas
        metric_name: Nombre de la métrica
        decimal_places: Número de decimales

    Returns:
        DataFrame formateado
    """
    formatted = table.copy()
    formatted = formatted.round(decimal_places)
    formatted.columns.name = f'{metric_name} por Horizonte'
    return formatted


def load_predictions(folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga predicciones y valores reales desde una carpeta de experimento.

    Args:
        folder: Carpeta del experimento

    Returns:
        Tupla (predicciones, valores_reales) o (None, None) si no existen
    """
    pred_file = folder / "pred.npy"
    true_file = folder / "true.npy"

    if pred_file.exists() and true_file.exists():
        preds = np.load(pred_file)
        trues = np.load(true_file)
        return preds, trues

    return None, None


def load_dataset(dataset_name: str, data_path: str = "../dataset/ETT-small") -> pd.DataFrame:
    """
    Carga un dataset ETT.

    Args:
        dataset_name: Nombre del dataset (ETTh1, ETTh2, ETTm1, ETTm2)
        data_path: Ruta al directorio de datos

    Returns:
        DataFrame con los datos
    """
    file_path = Path(data_path) / f"{dataset_name}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    return None
