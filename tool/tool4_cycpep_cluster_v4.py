#!/usr/bin/env python3
"""
Cyclic Peptide Clustering Tool with Data Export
===============================================
Clusters cyclic peptide sequences with data export for vector graphics

Author: Assistant
Date: 2026-04-09
"""

import numpy as np
import pandas as pd
import json
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Dict, Tuple, Optional
import os

warnings.filterwarnings('ignore')

def numpy_to_json_serializable(obj):
    """Convert NumPy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.intc, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

class AminoAcidProperties:
    """Amino acid properties based on scientific literature."""

    PROPERTIES = {
        'A': {'hydrophobicity': 1.8, 'charge': 0, 'aromaticity': 0,
              'polarity': -0.5, 'volume': 89, 'flexibility': 0.5},
        'R': {'hydrophobicity': -4.5, 'charge': 1, 'aromaticity': 0,
              'polarity': -3.5, 'volume': 170, 'flexibility': 0.2},
        'N': {'hydrophobicity': -3.5, 'charge': 0, 'aromaticity': 0,
              'polarity': -3.5, 'volume': 113, 'flexibility': 0.3},
        'D': {'hydrophobicity': -3.5, 'charge': -1, 'aromaticity': 0,
              'polarity': -3.5, 'volume': 115, 'flexibility': 0.3},
        'C': {'hydrophobicity': 2.5, 'charge': 0, 'aromaticity': 0,
              'polarity': -0.5, 'volume': 106, 'flexibility': 0.5},
        'Q': {'hydrophobicity': -3.5, 'charge': 0, 'aromaticity': 0,
              'polarity': -2.5, 'volume': 138, 'flexibility': 0.4},
        'E': {'hydrophobicity': -3.5, 'charge': -1, 'aromaticity': 0,
              'polarity': -3.5, 'volume': 135, 'flexibility': 0.4},
        'G': {'hydrophobicity': -0.4, 'charge': 0, 'aromaticity': 0,
              'polarity': -0.4, 'volume': 60, 'flexibility': 1.0},
        'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'aromaticity': 0.5,
              'polarity': -3.5, 'volume': 142, 'flexibility': 0.2},
        'I': {'hydrophobicity': 4.5, 'charge': 0, 'aromaticity': 0,
              'polarity': -4.5, 'volume': 163, 'flexibility': 0.1},
        'L': {'hydrophobicity': 3.8, 'charge': 0, 'aromaticity': 0,
              'polarity': -3.8, 'volume': 163, 'flexibility': 0.1},
        'K': {'hydrophobicity': -3.9, 'charge': 1, 'aromaticity': 0,
              'polarity': -3.9, 'volume': 169, 'flexibility': 0.2},
        'M': {'hydrophobicity': 1.9, 'charge': 0, 'aromaticity': 0,
              'polarity': -1.9, 'volume': 162, 'flexibility': 0.2},
        'F': {'hydrophobicity': 2.8, 'charge': 0, 'aromaticity': 1,
              'polarity': -2.8, 'volume': 193, 'flexibility': 0.1},
        'P': {'hydrophobicity': -1.6, 'charge': 0, 'aromaticity': 0,
              'polarity': -1.6, 'volume': 136, 'flexibility': 0.0},
        'S': {'hydrophobicity': -0.8, 'charge': 0, 'aromaticity': 0,
              'polarity': -0.8, 'volume': 88, 'flexibility': 0.6},
        'T': {'hydrophobicity': -0.7, 'charge': 0, 'aromaticity': 0,
              'polarity': -0.7, 'volume': 107, 'flexibility': 0.5},
        'W': {'hydrophobicity': -0.9, 'charge': 0, 'aromaticity': 1,
              'polarity': -0.9, 'volume': 253, 'flexibility': 0.1},
        'Y': {'hydrophobicity': -1.3, 'charge': 0, 'aromaticity': 1,
              'polarity': -1.3, 'volume': 214, 'flexibility': 0.2},
        'V': {'hydrophobicity': 4.2, 'charge': 0, 'aromaticity': 0,
              'polarity': -4.2, 'volume': 140, 'flexibility': 0.1},
    }

    PROPERTY_NAMES = ['hydrophobicity', 'charge', 'aromaticity', 'polarity', 'volume', 'flexibility']


class CyclicPeptideClusterer:
    """Cyclic peptide clustering with data export functionality."""

    def __init__(self, property_weights: List[float] = None,
                 priority_sequences: List[str] = None,
                 priority_threshold: float = 0.5):
        """Initialize the clusterer."""
        if property_weights is None:
            property_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.property_weights = np.array(property_weights) / np.sum(property_weights)
        self.priority_sequences = [seq.upper().replace(' ', '').replace('-', '')
                                   for seq in priority_sequences] if priority_sequences else []
        self.priority_threshold = priority_threshold
        self.sequences = []
        self.normalized_sequences = []
        self.feature_vectors = []
        self.cluster_labels = None
        self.linkage_matrix = None
        self.representative_sequences = []
        self.priority_cluster_indices = []
        self.distance_matrix = None

    def _normalize_cyclic_sequence(self, sequence: str) -> str:
        """Normalize cyclic sequence to canonical form."""
        sequence = sequence.upper().replace(' ', '').replace('-', '').replace('_', '')
        if not sequence:
            return ""
        n = len(sequence)
        rotations = [sequence[i:] + sequence[:i] for i in range(n)]
        return min(rotations)

    def _get_sequence_features(self, sequence: str) -> np.ndarray:
        """Convert sequence to feature vector."""
        if not sequence:
            return np.zeros(len(AminoAcidProperties.PROPERTY_NAMES))
        feature_vector = np.zeros(len(AminoAcidProperties.PROPERTY_NAMES))
        for aa in sequence:
            aa = aa.upper()
            if aa in AminoAcidProperties.PROPERTIES:
                feature_vector += np.array([AminoAcidProperties.PROPERTIES[aa][prop]
                                            for prop in AminoAcidProperties.PROPERTY_NAMES])
        feature_vector = feature_vector * self.property_weights
        feature_vector = feature_vector / len(sequence)
        return feature_vector

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        n = len(self.feature_vectors)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((self.feature_vectors[i] - self.feature_vectors[j]) ** 2))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def load_sequences_list(self, sequences: List[str]) -> None:
        """Load sequences from list."""
        self.sequences = sequences
        self.normalized_sequences = [self._normalize_cyclic_sequence(seq) for seq in sequences]

    def compute_features(self) -> None:
        """Compute feature vectors."""
        self.feature_vectors = np.array([self._get_sequence_features(seq)
                                         for seq in self.normalized_sequences])

    def _find_priority_sequence_in_cluster(self, cluster_indices: np.ndarray) -> Optional[int]:
        """Find priority sequence in a cluster."""
        if not self.priority_sequences:
            return None
        for idx in cluster_indices:
            norm_seq = self.normalized_sequences[idx]
            if norm_seq in self.priority_sequences:
                return idx
        return None

    def cluster(self, n_clusters: int = None,
                method: str = 'average',
                linkage_type: str = 'ward') -> None:
        """Perform hierarchical clustering."""
        if len(self.sequences) < 2:
            raise ValueError("Need at least 2 sequences for clustering")

        self.distance_matrix = self._calculate_distance_matrix()
        self.linkage_matrix = linkage(self.distance_matrix, method=linkage_type)

        if n_clusters is not None:
            self.cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        else:
            self.cluster_labels = None

        self._get_representative_sequences_with_priority()

    def _get_representative_sequences_with_priority(self) -> None:
        """Get representative sequences with priority support."""
        if self.cluster_labels is None:
            return

        unique_labels = np.unique(self.cluster_labels)
        self.representative_sequences = []
        self.priority_cluster_indices = []

        for label in unique_labels:
            indices = np.where(self.cluster_labels == label)[0]
            cluster_features = self.feature_vectors[indices]
            centroid = np.mean(cluster_features, axis=0)

            priority_idx = self._find_priority_sequence_in_cluster(indices)

            if priority_idx is not None:
                priority_feature = self.feature_vectors[priority_idx]
                priority_distance = np.sqrt(np.sum((priority_feature - centroid) ** 2))
                max_distance = np.max([np.sqrt(np.sum((f - centroid) ** 2)) for f in cluster_features])

                if priority_distance <= self.priority_threshold * max_distance:
                    self.representative_sequences.append({
                        'cluster': label,
                        'sequence': self.normalized_sequences[priority_idx],
                        'original': self.sequences[priority_idx],
                        'index': priority_idx,
                        'is_priority': True,
                        'distance_to_centroid': priority_distance
                    })
                    self.priority_cluster_indices.append(label)
                else:
                    distances = np.sqrt(np.sum((cluster_features - centroid) ** 2, axis=1))
                    closest_idx = indices[np.argmin(distances)]
                    self.representative_sequences.append({
                        'cluster': label,
                        'sequence': self.normalized_sequences[closest_idx],
                        'original': self.sequences[closest_idx],
                        'index': closest_idx,
                        'is_priority': False,
                        'distance_to_centroid': distances[np.argmin(distances)]
                    })
            else:
                distances = np.sqrt(np.sum((cluster_features - centroid) ** 2, axis=1))
                closest_idx = indices[np.argmin(distances)]
                self.representative_sequences.append({
                    'cluster': label,
                    'sequence': self.normalized_sequences[closest_idx],
                    'original': self.sequences[closest_idx],
                    'index': closest_idx,
                    'is_priority': False,
                    'distance_to_centroid': distances[np.argmin(distances)]
                })

    def export_cluster_data(self, export_dir: str = 'export_data') -> Dict[str, str]:
        """
        Export all clustering data to CSV files.

        Args:
            export_dir: Directory to save exported files

        Returns:
            Dictionary of file paths
        """
        os.makedirs(export_dir, exist_ok=True)
        file_paths = {}

        # 1. Export sequences and features
        sequences_df = pd.DataFrame({
            'Index': range(len(self.sequences)),
            'Original_Sequence': self.sequences,
            'Normalized_Sequence': self.normalized_sequences,
            'Cluster_ID': self.cluster_labels,
            'Hydrophobicity': self.feature_vectors[:, 0],
            'Charge': self.feature_vectors[:, 1],
            'Aromaticity': self.feature_vectors[:, 2],
            'Polarity': self.feature_vectors[:, 3],
            'Volume': self.feature_vectors[:, 4],
            'Flexibility': self.feature_vectors[:, 5]
        })
        seq_file = os.path.join(export_dir, 'sequences_and_features.csv')
        sequences_df.to_csv(seq_file, index=False)
        file_paths['sequences'] = seq_file

        # 2. Export cluster summary
        summary_df = self._get_cluster_summary_df()
        summary_file = os.path.join(export_dir, 'cluster_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        file_paths['summary'] = summary_file

        # 3. Export distance matrix
        if self.distance_matrix is not None:
            dist_df = pd.DataFrame(self.distance_matrix,
                                   index=[f'S{i + 1}' for i in range(len(self.sequences))],
                                   columns=[f'S{i + 1}' for i in range(len(self.sequences))])
            dist_file = os.path.join(export_dir, 'distance_matrix.csv')
            dist_df.to_csv(dist_file)
            file_paths['distance_matrix'] = dist_file

        # 4. Export linkage matrix (for dendrogram)
        if self.linkage_matrix is not None:
            linkage_df = pd.DataFrame(self.linkage_matrix,
                                      columns=['Index1', 'Index2', 'Distance', 'Count'])
            linkage_file = os.path.join(export_dir, 'linkage_matrix.csv')
            linkage_df.to_csv(linkage_file, index=False)
            file_paths['linkage'] = linkage_file

        # 5. Export representative sequences
        rep_data = []
        for rep in self.representative_sequences:
            rep_data.append({
                'Cluster_ID': rep['cluster'],
                'Sequence': rep['sequence'],
                'Original': rep['original'],
                'Is_Priority': rep['is_priority'],
                'Distance_to_Centroid': rep['distance_to_centroid']
            })
        rep_df = pd.DataFrame(rep_data)
        rep_file = os.path.join(export_dir, 'representative_sequences.csv')
        rep_df.to_csv(rep_file, index=False)
        file_paths['representatives'] = rep_file

        # 6. Export property data for bar chart
        property_data = self._get_property_data_for_export()
        prop_df = pd.DataFrame(property_data)
        prop_file = os.path.join(export_dir, 'property_comparison_data.csv')
        prop_df.to_csv(prop_file, index=False)
        file_paths['properties'] = prop_file

        # 7. Export as JSON (for web/interactive use)
        json_data = {
            'sequences': self.sequences,
            'normalized_sequences': self.normalized_sequences,
            'feature_vectors': numpy_to_json_serializable(self.feature_vectors),
            'cluster_labels': numpy_to_json_serializable(self.cluster_labels),
            'linkage_matrix': numpy_to_json_serializable(
                self.linkage_matrix) if self.linkage_matrix is not None else None,
            'distance_matrix': numpy_to_json_serializable(
                self.distance_matrix) if self.distance_matrix is not None else None,
            'representative_sequences': numpy_to_json_serializable(self.representative_sequences),
            'priority_sequences': self.priority_sequences,
            'property_weights': numpy_to_json_serializable(self.property_weights)
        }
        json_file = os.path.join(export_dir, 'clustering_data.json')
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        file_paths['json'] = json_file

        # 8. Export similarity matrix (for heatmap)
        if self.distance_matrix is not None:
            max_dist = np.max(self.distance_matrix)
            similarity_matrix = 1 - self.distance_matrix / max_dist
            sim_df = pd.DataFrame(similarity_matrix,
                                  index=[f'S{i + 1}' for i in range(len(self.sequences))],
                                  columns=[f'S{i + 1}' for i in range(len(self.sequences))])
            sim_file = os.path.join(export_dir, 'similarity_matrix.csv')
            sim_df.to_csv(sim_file)
            file_paths['similarity'] = sim_file

        print(f"\nData exported to: {export_dir}/")
        for name, path in file_paths.items():
            print(f"  - {name}: {path}")

        return file_paths

    def _get_cluster_summary_df(self) -> pd.DataFrame:
        """Get cluster summary as DataFrame."""
        labels = self.cluster_labels
        unique_clusters = np.unique(labels)

        # Reorder clusters (priority first)
        priority_clusters = [c for c in unique_clusters if c in self.priority_cluster_indices]
        non_priority_clusters = [c for c in unique_clusters if c not in self.priority_cluster_indices]
        ordered_clusters = priority_clusters + non_priority_clusters

        summary = []
        for cluster in ordered_clusters:
            indices = np.where(labels == cluster)[0]
            cluster_size = len(indices)
            centroid = np.mean(self.feature_vectors[indices], axis=0)

            summary.append({
                'Cluster_ID': cluster,
                'Size': cluster_size,
                'Representative': self.representative_sequences[cluster - 1]['sequence'] if cluster <= len(
                    self.representative_sequences) else '',
                'Is_Priority': cluster in self.priority_cluster_indices,
                'Centroid_Hydrophobicity': centroid[0],
                'Centroid_Charge': centroid[1],
                'Centroid_Aromaticity': centroid[2],
                'Centroid_Polarity': centroid[3],
                'Centroid_Volume': centroid[4],
                'Centroid_Flexibility': centroid[5]
            })

        return pd.DataFrame(summary)

    def _get_property_data_for_export(self) -> List[Dict]:
        """Get property comparison data for bar chart."""
        if self.cluster_labels is None:
            return []

        labels = self.cluster_labels
        unique_clusters = np.unique(labels)
        priority_clusters = [c for c in unique_clusters if c in self.priority_cluster_indices]
        non_priority_clusters = [c for c in unique_clusters if c not in self.priority_cluster_indices]
        ordered_clusters = priority_clusters + non_priority_clusters

        property_data = []
        for label in ordered_clusters:
            indices = np.where(labels == label)[0]
            cluster_features = self.feature_vectors[indices]

            for i, prop in enumerate(AminoAcidProperties.PROPERTY_NAMES):
                property_data.append({
                    'Cluster_ID': label,
                    'Property': prop,
                    'Mean': np.mean(cluster_features[:, i]),
                    'Std': np.std(cluster_features[:, i]),
                    'Is_Priority': label in self.priority_cluster_indices
                })

        return property_data

    def save_vector_graphics(self, save_dir: str = 'vector_graphics') -> Dict[str, str]:
        """
        Save all plots as vector graphics (SVG/PDF).

        Args:
            save_dir: Directory to save vector graphics

        Returns:
            Dictionary of file paths
        """
        os.makedirs(save_dir, exist_ok=True)
        file_paths = {}

        # 1. Dendrogram
        fig1, ax = plt.subplots(figsize=(14, 10))
        labels = []
        for i in range(len(self.sequences)):
            label = f'S{i + 1}'
            if self.normalized_sequences[i] in self.priority_sequences:
                label = f'★{label}★'
            labels.append(label)

        dendrogram(self.linkage_matrix,
                   labels=labels,
                   leaf_rotation=90,
                   ax=ax,
                   show_contracted=False)
        ax.set_title('Cyclic Peptide Clustering Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sequence Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        plt.tight_layout()

        dendro_svg = os.path.join(save_dir, 'dendrogram.svg')
        fig1.savefig(dendro_svg, dpi=300, format='svg')
        file_paths['dendrogram_svg'] = dendro_svg

        dendro_pdf = os.path.join(save_dir, 'dendrogram.pdf')
        fig1.savefig(dendro_pdf, dpi=300, format='pdf', bbox_inches='tight')
        file_paths['dendrogram_pdf'] = dendro_pdf
        plt.close(fig1)

        # 2. Heatmap
        fig2, ax = plt.subplots(figsize=(14, 12))
        if self.distance_matrix is not None:
            max_dist = np.max(self.distance_matrix)
            similarity_matrix = 1 - self.distance_matrix / max_dist
            labels = []
            for i in range(len(self.sequences)):
                short_seq = self.normalized_sequences[i][:10]
                if len(self.normalized_sequences[i]) > 10:
                    short_seq += '...'
                if self.normalized_sequences[i] in self.priority_sequences:
                    labels.append(f'★S{i + 1}★\n{short_seq}')
                else:
                    labels.append(f'S{i + 1}\n{short_seq}')

            sns.heatmap(similarity_matrix,
                        annot=False,
                        cmap='viridis',
                        center=0.5,
                        ax=ax,
                        cbar_kws={'label': 'Similarity'})

            ax.set_title('Cyclic Peptide Clustering Heatmap', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Normalized Sequence', fontsize=12)
            ax.set_ylabel('Normalized Sequence', fontsize=12)

        plt.tight_layout()

        heatmap_svg = os.path.join(save_dir, 'heatmap.svg')
        fig2.savefig(heatmap_svg, dpi=300, format='svg')
        file_paths['heatmap_svg'] = heatmap_svg

        heatmap_pdf = os.path.join(save_dir, 'heatmap.pdf')
        fig2.savefig(heatmap_pdf, dpi=300, format='pdf', bbox_inches='tight')
        file_paths['heatmap_pdf'] = heatmap_pdf
        plt.close(fig2)

        # 3. Property comparison
        fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        property_names = AminoAcidProperties.PROPERTY_NAMES
        unique_clusters = np.unique(self.cluster_labels) if self.cluster_labels is not None else []
        priority_clusters = [c for c in unique_clusters if c in self.priority_cluster_indices]
        non_priority_clusters = [c for c in unique_clusters if c not in self.priority_cluster_indices]
        ordered_clusters = priority_clusters + non_priority_clusters

        for i, prop in enumerate(property_names):
            ax = axes[i]
            cluster_means = []
            cluster_stds = []

            for label in ordered_clusters:
                indices = np.where(self.cluster_labels == label)[0]
                prop_values = self.feature_vectors[indices, i]
                cluster_means.append(np.mean(prop_values))
                cluster_stds.append(np.std(prop_values))

            colors = ['red' if label in self.priority_cluster_indices else 'blue'
                      for label in ordered_clusters]

            ax.bar(range(len(ordered_clusters)), cluster_means,
                   yerr=cluster_stds,
                   capsize=10,
                   color=colors,
                   edgecolor='black')

            ax.set_xlabel('Cluster ID', fontsize=12)
            ax.set_ylabel(prop.capitalize(), fontsize=12)
            ax.set_title(f'{prop.capitalize()} Distribution', fontsize=14)
            ax.set_xticks(range(len(ordered_clusters)))
            ax.set_xticklabels(ordered_clusters)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Amino Acid Property Comparison Across Clusters', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        prop_svg = os.path.join(save_dir, 'property_comparison.svg')
        fig3.savefig(prop_svg, dpi=300, format='svg')
        file_paths['property_svg'] = prop_svg

        prop_pdf = os.path.join(save_dir, 'property_comparison.pdf')
        fig3.savefig(prop_pdf, dpi=300, format='pdf', bbox_inches='tight')
        file_paths['property_pdf'] = prop_pdf
        plt.close(fig3)

        print(f"\nVector graphics saved to: {save_dir}/")
        for name, path in file_paths.items():
            print(f"  - {name}: {path}")

        return file_paths


def create_demo_sequences() -> List[str]:
    """Create demo cyclic peptide sequences."""
    return [
        "KRQQV",
        "PVWQY",
        "PRRVY",
        "PRWPV",
        "PRWPW",
        "KRQWV",
        "PQWQY",
        "KRMWV",
        "KKMQV",
        "KKMYV",
        "KKMLV",
        "KKMMV",
        "PRRPW",
        "PSWRV",
        "PTWRV",
        "PRYWW",
        "KKMVV",
        "PTRRV",
        "KRQWW",
        "KRMWW",
        "KKMQW",
        "PQYQY",
        "PVYQY",
        "PRWWW",
        "KKMMW",
        "KRQVY",
        "PQWRV",
        "PVWRW",
        "KKMLW",
        "PTYRV",
        "PRWVY",
        "PSYRV",
        "KKQMV",
        "PSWRW",
        "KKQMQ",
        "KRQYV",
        "KKMVW",
        "KKQMW",
        "PTWRW",
        "KRMYV",
        "PRRWV",
        "KRMNV",
        "PRYWV",
        "PVRRW",
        "PVYRW",
        "KRMNW",
        "PSQRV",
        "PQYRV",
        "PSRRV",
        "PSRRW",
        "PQRRQ",
        "PVQRV",
        "PRQWW",
        "PQWRW",
        "KRMYW",
        "KRMMV",
        "KRQVV",
        "KRQLV",
        "PTQRV",
        "PSYRW",
        "PRWYW",
        "PVRRV",
        "PTRRW",
        "PRYPW",
        "PVSQY",
        "KRMQV",
        "PRRWW",
        "PRPRV",
        "PTYRW",
        "KRMQW",
        "KRQMV",
        "KRMSV",
        "PSSRV",
        "PRYYW",
        "PRSWW",
        "KRMSW",
        "KRQNV",
        "PRRYW",
        "KRQMW",
        "PTSRV",
        "KRMLV",
        "PQRRV",
        "PRSPV",
        "KRMTV",
        "PQYRW",
        "KRQSV",
        "PRYPV",
        "PRPVY",
        "PRWWV",
        "PSWRT",
        "KRMVV",
        "PRTPV",
        "KRQTV",
        "PVTQY",
        "PVQRW",
        "PQQRQ",
        "PRQVY",
        "PRTWW",
        "KMQRV",
        "KSMRV",
        "KRMMW",
        "PVVQY",
        "PRRYV",
        "KRMLW",
        "PSTRV",
        "PTWQY",
        "PRVPV",
        "PSQRW",
        "PTTRV",
        "PRQYW",
        "KRQLW",
        "KLMRV",
        "PSQRT",
        "PQQRV",
        "KSQRV",
        "PQWRT",
        "KVMRV",
        "PRYWY",
        "KWQQY",
        "KRMVW",
        "PRWWY",
        "PQRRW",
        "PRSPW",
        "PRRWY",
        "PVRQY",
        "KKQLV",
        "PVSRW",
        "KLQRV",
        "KNQRV",
        "PTQRW",
        "PRTVY",
        "KTMRV",
        "KKQLW",
        "PTVRV",
        "PQSQY",
        "KRMTW",
        "PRYYV",
        "KVQRV",
        "PRSVY",
        "PYQYY",
        "KLQRW",
        "PRVWW",
        "PRQWV",
        "KMMRV",
        "KYMQY",
        "PSSRW",
        "KQQRV",
        "PRPVV",
        "PVVRW",
        "PQTQY",
        "PRPTV",
        "PSWQY",
        "KKMMP",
        "KSMRW",
        "PSVRV",
        "KRQVW",
        "KKQLY",
        "KVMRW",
        "KTQRV",
        "PRWYV",
        "PRRRY",
        "KTMRW",
        "PQVQY",
        "KVQRW",
        "PRQWY",
        "PTRRT",
        "PVTRW",
        "KKMQP",
        "KWQRW",
        "KKQVV",
        "PSYRT",
        "PSTRW",
        "KMQRW",
        "PRRSY",
        "PWWQY",
        "PRRTY",
        "KKMRV",
        "PWYQY",
        "PSRRT",
        "PRRSV",
        "PTSRW",
        "PWRQY",
        "KKQWQ",
        "KKQVW",
        "KLMRW",
        "PRRWT",
        "PRVVY",
        "PWRRW",
        "PQQRW",
        "PRTPW",
        "PSVRW",
        "PTYQY",
        "PRWVW",
        "PRWSW",
        "PQRQY",
        "PRPWY",
        "KKMRQ",
        "KMQRQ",
        "PWQQY",
        "PRPSV",
        "PTTRW",
        "PQSRV",
        "KMMRW",
        "KLQRQ",
        "KRQWT",
        "KRQTW",
        "PRVPW",
        "PQYRT",
        "PRWTW",
        "PTQRT",
        "PQTRV",
        "PQWRS",
        "PQVRV",
        "PTVRW",
        "PRRVV",
        "PRSYW",
        "PRSWV",
        "KKMLP",
        "PRRTV",
        "PRWWT",
        "KRMWT",
        "PRSWY",
        "PRYWT",
        "PRVYW",
        "PVQQY",
        "KKQRV",
        "PSQRS",
        "KKQLQ",
        "KPRPQ",
        "KKMMT",
        "PRQYV",
        "KKMLY",
        "PRYSW",
        "PRRQV",
        "PRYVW",
        "KKQQV",
        "PQWRR",
        "PRTYW",
        "KKMMM",
        "PRRWS",
        "PRWQW",
        "KKMVP",
        "PRYTW",
        "PRVWV",
        "PTRQY",
        "KMQRP",
        "PRTWY",
        "PRTWV",
        "PRPRW",
        "KKMQT",
        "PYQQY",
        "PSRRS",
        "ACRYI",
        "PRVWY",
        "PSYQY",
        "KLMRP",
        "PWQRW",
        "KRQPV",
        "PRYQW",
        "KKMPY",
        "KMMRP",
        "FPKPQ",
        "KRQPY",
        "KMQKW",
        "KRQWR",
        "PRWWS",
        "KRMPV",
        "PRRVW",
        "PSRQY",
        "PRRYT",
        "KKQVQ",
        "PRYWS",
        "PRRYS",
        "PQVRW",
        "PQRRT",
        "KKMYM",
        "APGKQ",
        "PQSRW",
        "KKMQN",
        "KRRWV",
        "KLMKV",
        "PQRRS",
        "KMMKV",
        "KKMRW",
        "KKMYR",
        "PRWWR",
        "PQTRW",
        "PRPWW",
        "KMQQY",
        "KLQRP",
        "PRWYT",
        "PRYSV",
        "KLMKW",
        "KRMWR",
        "PQVYY",
        "KRMPW",
        "PRYVV",
        "PRWSV",
        "KKQMT",
        "PQQRT",
        "PRWVV",
        "KKQMR",
        "KKMMR",
        "KMQRM",
        "ARYKV",
        "PRQSW",
        "KRMWS",
        "PSQQY",
        "KKMKY",
        "KRQWS",
        "PRRSW",
        "PTQQY",
        "PRRQW",
        "KRMRV",
        "KKMLN",
        "PRQWT",
        "PQQQY",
        "PRWYS",
        "KKMMN",
        "KRQPW",
        "PRQTW",
        "KLMRM",
        "PRYTV",
        "PRSWT",
        "PRQVW",
        "PRPRS",
        "PRTYV",
        "KKMMS",
        "KKMYS",
        "ARWKV",
        "HPKPQ",
        "PRQQW",
        "PRTWT",
        "PRRYR",
        "KLQRL",
        "PQWYY",
        "PRQSY",
        "PRQRY",
        "PRSYV",
        "PRQTY",
        "PRPVW",
        "KNVRP",
        "KRMMT",
        "PRYVY",
        "KMMRM",
        "PRWTV",
        "KKQMS",
        "PQYRS",
        "PRVYV",
        "KKMQM",
        "KRMYT",
        "KRQRV",
        "KRQWY",
        "KRRWW",
        "PRWPT",
        "KPKPQ",
        "PRYQV",
        "KKMLT",
        "PSVRT",
        "KKMTY",
        "PRSVW",
        "KRRVY",
        "KKMVN",
        "PRSTW",
        "PRQWS",
        "ARQKV",
        "KKMVT",
        "KMMQY",
        "PRSWS",
        "PRSSW",
        "PRQVV",
        "PQRRR",
        "PRTWS",
        "KLQRM",
        "KKMQS",
        "KKMRP",
        "PSSRT",
        "KLQQY",
        "GRRMV",
        "KKMLM",
        "PSTRT",
        "PRPSW",
        "KRMQT",
        "KMQRS",
        "PRQSV",
        "PRWQV",
        "KRMQS",
        "KKMLS",
        "KKMVM",
        "KKMTV",
        "PRQTV",
        "PRTWR",
        "KLMQY",
        "KRMYR",
        "PRPYY",
        "PRWYR",
        "KLMRL",
        "PWVQY",
        "PQQRS",
        "KRMRW"
    ]


def main():
    """Main function with data export."""
    print("=" * 80)
    print("CYCLIC PEPTIDE CLUSTERING TOOL WITH DATA EXPORT")
    print("=" * 80)

    # Create clusterer with priority sequences
    property_weights = [1.5, 1.5, 1.0, 1.0, 1.0, 1.0]
    priority_sequences = ["KRQQV","PSQRV","PVWRW","PRYSW","KMMKV"]

    clusterer = CyclicPeptideClusterer(
        property_weights=property_weights,
        priority_sequences=priority_sequences,
        priority_threshold=0.5
    )

    # Load sequences
    sequences = create_demo_sequences()
    clusterer.load_sequences_list(sequences)

    print(f"\nLoaded {len(sequences)} sequences")
    print(f"Priority sequences: {priority_sequences}")

    # Compute features and cluster
    clusterer.compute_features()
    clusterer.cluster(n_clusters=5, method='average', linkage_type='ward')

    # Export data
    print("\n" + "=" * 80)
    print("EXPORTING DATA FOR VECTOR GRAPHICS")
    print("=" * 80)

    # 1. Export raw data (CSV/JSON)
    data_files = clusterer.export_cluster_data(export_dir='export_data')

    # 2. Save vector graphics (SVG/PDF)
    vector_files = clusterer.save_vector_graphics(save_dir='vector_graphics')

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print("\nFor vector graphics, use:")
    print("  - SVG files: Open in Inkscape, Adobe Illustrator, or browser")
    print("  - PDF files: Open in Adobe Acrobat, Inkscape, or LaTeX")
    print("\nFor custom editing, use exported CSV files:")
    print("  - Import into Adobe Illustrator, Excel, or your preferred tool")
    print("  - Re-create charts with your own styling")


if __name__ == "__main__":
    main()
