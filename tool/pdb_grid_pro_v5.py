#!/usr/bin/env python3
"""
PDB Grid Pro - Merged Edition

Updates:
- Replaced static reference-only region definition with global PCA projection and gridding (from V2).
- Bridged V2 spatial splitting with V4 position-specific statistical testing (1-sample t-test).
- Added explicit export of the amino acid residues contained in each spatial split.
"""

import sys
import argparse
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import PDB
from sklearn.decomposition import PCA
from scipy import stats
from collections import defaultdict
import warnings
import tempfile

# Suppress Biopython PDB warnings
warnings.simplefilter('ignore', PDB.PDBExceptions.PDBConstructionWarning)

# Set up debug logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- EXPERIMENTAL DATASETS ---
HESSA_SCALE = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': 0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}
PKA_IPC2 = {
    'ALA': 8.1, 'ARG': 10.5, 'ASN': 11.6, 'ASP': 13.0, 'CYS': 5.5,
    'GLN': 10.5, 'GLU': 12.3, 'GLY': 9.0, 'HIS': 10.4, 'ILE': 5.2,
    'LEU': 4.9, 'LYS': 11.3, 'MET': 5.7, 'PHE': 5.2, 'PRO': 8.0,
    'SER': 9.2, 'THR': 8.6, 'TRP': 5.4, 'TYR': 6.2, 'VAL': 5.9
}


class PDBGridPro:
    def __init__(self, pdb_path, grid_size=20.0, ph=7.4, ref_model_idx=8, weights=(1.0, 1.0, 1.0), sim_mode=False):
        self.pdb_path = pdb_path
        self.grid_size = grid_size
        self.ph = ph
        self.ref_model_idx = ref_model_idx
        self.weights = {'hydro': weights[0], 'rmsd': weights[1], 'charge': weights[2]}
        self.sim_mode = sim_mode

        self.parser = PDB.PDBParser(QUIET=True)
        self.models = []

        # Datastructures to handle merged V2/V4 logic
        self.projected_data = []
        self.grid_cells = defaultdict(lambda: defaultdict(list))
        self.regions = defaultdict(list)
        self.scores_df = None
        self.time_series_df = None

    def run_pipeline(self, output_prefix):
        self.load_structure()
        self.perform_pca_and_project()  # From V2
        self.partition_into_grid()  # From V2
        self.compute_per_model_metrics()  # From V4 (Updated to track residues)
        self.normalize_and_score()
        self.calculate_time_series_stats(output_prefix)
        self.export_results(output_prefix)
        self.plot_results(output_prefix)

    def load_structure(self):
        logging.info(f"Loading PDB: {self.pdb_path}")
        structure = self.parser.get_structure('struct', self.pdb_path)
        self.models = list(structure.get_models())
        logging.info(f"Found {len(self.models)} models.")

        if self.ref_model_idx >= len(self.models) or self.ref_model_idx < 0:
            raise ValueError(f"Reference model index {self.ref_model_idx} is out of bounds. "
                             f"PDB contains {len(self.models)} models (0-indexed).")
        logging.info(f"Using model {self.ref_model_idx} as the structural and sequence reference.")

    def perform_pca_and_project(self):
        """Perform PCA globally across all models to define dynamic bounds."""
        logging.info("Performing PCA on C-alpha coordinates...")
        all_coords = []
        metadata = []

        for m_idx, model in enumerate(self.models):
            for chain in model:
                for res in chain:
                    if 'CA' in res and PDB.is_aa(res):
                        ca = res['CA']
                        coord = ca.get_coord()
                        all_coords.append(coord)
                        metadata.append({
                            'model_idx': m_idx, 'chain': chain.id,
                            'res_seq': res.id[1], 'res_name': res.get_resname(),
                            'coord_3d': coord
                        })

        if not all_coords:
            raise ValueError("No CA coordinates found in the PDB file.")

        X = np.array(all_coords)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        for i, meta in enumerate(metadata):
            meta['coord_2d'] = X_2d[i]
            self.projected_data.append(meta)

    def partition_into_grid(self):
        """Partitions globally projected coordinates into spatial regions."""
        logging.info("Partitioning structures into grid cells...")
        coords_2d = np.array([m['coord_2d'] for m in self.projected_data])
        min_u, min_v = coords_2d.min(axis=0)

        self.regions.clear()

        for item in self.projected_data:
            u, v = item['coord_2d']
            # Format to match V4 Region ID format
            grid_id = f"{int((u - min_u) // self.grid_size)}_{int((v - min_v) // self.grid_size)}"
            item['grid_id'] = grid_id
            self.grid_cells[grid_id][item['model_idx']].append(item)

            # Bridge to V4 logic: Populate self.regions using the reference model's baseline
            if item['model_idx'] == self.ref_model_idx:
                self.regions[grid_id].append((item['chain'], item['res_seq']))

        logging.info(f"Partitioned structure into {len(self.grid_cells)} dynamic spatial regions.")

    def calculate_time_series_stats(self, prefix):
        logging.info("Calculating per-position time-series statistics...")
        ref_model = self.models[self.ref_model_idx]
        ts_data = []

        for chain in ref_model:
            for res in chain:
                if 'CA' in res and PDB.is_aa(res):
                    ch_id = chain.id
                    res_seq = res.id[1]
                    ref_res_name = res.get_resname()

                    ref_hydro = HESSA_SCALE.get(ref_res_name, 0.0)
                    ref_charge = PKA_IPC2.get(ref_res_name, 0.0)

                    row = {
                        'Chain': ch_id, 'ResSeq': res_seq,
                        'Ref_ResName': ref_res_name,
                        'Ref_Hydro': ref_hydro, 'Ref_Charge': ref_charge
                    }

                    h_diffs, c_diffs = [], []

                    for m_idx, model in enumerate(self.models):
                        if m_idx == self.ref_model_idx: continue

                        try:
                            target_res = model[ch_id][res_seq]
                            target_res_name = target_res.get_resname()
                            t_hydro = HESSA_SCALE.get(target_res_name, 0.0)
                            t_charge = PKA_IPC2.get(target_res_name, 0.0)

                            h_diff = t_hydro - ref_hydro
                            c_diff = t_charge - ref_charge

                            row[f'M{m_idx}_Hydro_Diff'] = h_diff
                            row[f'M{m_idx}_Charge_Diff'] = c_diff
                            h_diffs.append(h_diff)
                            c_diffs.append(c_diff)

                        except KeyError:
                            row[f'M{m_idx}_Hydro_Diff'] = np.nan
                            row[f'M{m_idx}_Charge_Diff'] = np.nan

                    if len(h_diffs) > 1:
                        h_pval = stats.ttest_1samp(h_diffs, 0.0)[1] if np.std(h_diffs) > 0 else 1.0
                        c_pval = stats.ttest_1samp(c_diffs, 0.0)[1] if np.std(c_diffs) > 0 else 1.0

                        row['Mean_Hydro_Diff'] = np.mean(h_diffs)
                        row['Hydro_P_Value'] = h_pval
                        row['Sig_Hydro_Dev'] = h_pval < 0.05
                        row['Mean_Charge_Diff'] = np.mean(c_diffs)
                        row['Charge_P_Value'] = c_pval
                        row['Sig_Charge_Dev'] = c_pval < 0.05
                    else:
                        row['Hydro_P_Value'] = np.nan
                        row['Charge_P_Value'] = np.nan
                        row['Sig_Hydro_Dev'] = False
                        row['Sig_Charge_Dev'] = False

                    ts_data.append(row)

        self.time_series_df = pd.DataFrame(ts_data)
        out_path = f"{prefix}_timeseries_stats.csv"
        self.time_series_df.to_csv(out_path, index=False)
        logging.info(
            f"Detected {self.time_series_df['Sig_Hydro_Dev'].sum()} positions with significant Hydrophobicity deviations.")

    def compute_per_model_metrics(self):
        logging.debug("Computing per-residue differences and aggregating by region...")
        results = []
        ref_model = self.models[self.ref_model_idx]

        for grid_id, sequence_segment in self.regions.items():
            # Format the output of which amino acids are mapped to this split
            residue_list_str = ";".join([f"{ch}{seq}" for ch, seq in sequence_segment])

            for m_idx, target_model in enumerate(self.models):
                if m_idx == self.ref_model_idx:
                    continue

                sq_dists, hydro_diffs, charge_diffs = [], [], []

                for chain_id, res_seq in sequence_segment:
                    try:
                        ref_res = ref_model[chain_id][res_seq]
                        target_res = target_model[chain_id][res_seq]

                        ref_coord = ref_res['CA'].get_coord()
                        target_coord = target_res['CA'].get_coord()
                        sq_dists.append(np.sum((target_coord - ref_coord) ** 2))

                        ref_res_name = ref_res.get_resname()
                        target_res_name = target_res.get_resname()

                        ref_h = HESSA_SCALE.get(ref_res_name, 0.0)
                        tar_h = HESSA_SCALE.get(target_res_name, 0.0)
                        hydro_diffs.append(abs(tar_h - ref_h))

                        ref_c = PKA_IPC2.get(ref_res_name, 0.0)
                        tar_c = PKA_IPC2.get(target_res_name, 0.0)
                        charge_diffs.append(abs(tar_c - ref_c))

                    except KeyError:
                        continue

                if sq_dists:
                    rmsd = np.sqrt(np.mean(sq_dists))
                    mean_h_diff = np.mean(hydro_diffs)
                    mean_c_diff = np.mean(charge_diffs)
                else:
                    rmsd, mean_h_diff, mean_c_diff = np.nan, np.nan, np.nan

                results.append({
                    'Region': grid_id,
                    'Target_Model': m_idx,
                    'RMSD': rmsd,
                    'Hydro_Diff': mean_h_diff,
                    'Charge_Diff': mean_c_diff,
                    'Residue_Count': len(sq_dists),
                    'Residue_List': residue_list_str  # ADDED Tracking here
                })

        self.scores_df = pd.DataFrame(results).dropna()

    def normalize_and_score(self):
        df = self.scores_df
        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)

        df['n_RMSD'] = norm(df['RMSD'])
        df['n_Hydro'] = norm(df['Hydro_Diff'])
        df['n_Charge'] = norm(df['Charge_Diff'])

        df['Composite_Score'] = (
                self.weights['rmsd'] * df['n_RMSD'] +
                self.weights['hydro'] * df['n_Hydro'] +
                self.weights['charge'] * df['n_Charge']
        )

        if self.sim_mode:
            max_possible = sum(self.weights.values())
            df['Composite_Score'] = max_possible - df['Composite_Score']

        self.scores_df = df.sort_values(by=['Composite_Score'], ascending=False)

    def export_results(self, prefix):
        csv_path = f"{prefix}_region_scores.csv"
        self.scores_df.to_csv(csv_path, index=False)

        pivot_df = self.scores_df.pivot(index='Region', columns='Target_Model', values='Composite_Score').fillna(0)
        np.save(f"{prefix}_score_matrix.npy", pivot_df.values)

        # Include Residue_List in the summary file for clean tracking
        summary = self.scores_df.groupby('Region').agg({
            'Composite_Score': 'mean',
            'Residue_List': 'first'  # Grabs the mapped residues string
        }).reset_index()

        summary = summary.sort_values(by='Composite_Score', ascending=not self.sim_mode)
        summary.to_csv(f"{prefix}_summary_rankings.csv", index=False)

        logging.info(f"Regional Results exported to {csv_path}, .npy, and summary_rankings.csv")

    def plot_results(self, prefix):
        logging.debug("Generating output plots...")
        pivot_df = self.scores_df.pivot(index='Region', columns='Target_Model', values='Composite_Score').fillna(0)

        plt.figure(figsize=(10, 8))
        plt.imshow(pivot_df.values, aspect='auto', cmap='viridis')
        plt.colorbar(label='Composite Score')
        plt.yticks(ticks=np.arange(len(pivot_df.index)), labels=pivot_df.index, fontsize=8)
        plt.xticks(ticks=np.arange(len(pivot_df.columns)), labels=pivot_df.columns)
        plt.xlabel('Target Model Index')
        plt.ylabel('Spatial/Sequence Region ID')
        score_type = "Similarity" if self.sim_mode else "Difference"
        plt.title(f'Region vs Model Composite {score_type} Heatmap (Ref: {self.ref_model_idx})')
        plt.tight_layout()
        plt.savefig(f"{prefix}_heatmap.png")
        plt.close()

        summary = self.scores_df.groupby('Region')['Composite_Score'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        summary.plot(kind='bar', color='coral' if not self.sim_mode else 'skyblue', edgecolor='black')
        plt.ylabel('Average Composite Score')
        plt.title(f'Top 10 Regions with Highest {score_type} (vs Ref {self.ref_model_idx})')
        plt.tight_layout()
        plt.savefig(f"{prefix}_barplot.png")
        plt.close()


# --- UNIT TESTS ---
def run_tests():
    logging.getLogger().setLevel(logging.WARNING)
    print("\n--- Running Unit Tests ---")

    pdb_str = """MODEL        0
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C  
ENDMDL
MODEL        1
ATOM      1  N   VAL A   1      10.000  10.000  10.000  1.00 20.00           N  
ATOM      2  CA  VAL A   1      15.000  15.000  15.000  1.00 20.00           C  
ATOM      3  C   VAL A   1      16.000  16.000  16.000  1.00 20.00           C  
ENDMDL
MODEL        2
ATOM      1  N   ASP A   1      10.000  10.000  10.000  1.00 20.00           N  
ATOM      2  CA  ASP A   1      15.000  15.000  15.000  1.00 20.00           C  
ATOM      3  C   ASP A   1      16.000  16.000  16.000  1.00 20.00           C  
ENDMDL
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_str)
        temp_pdb = f.name

    try:
        app_stats = PDBGridPro(temp_pdb, ref_model_idx=0)
        app_stats.run_pipeline('test_merged')

        # Test 1: Verify statistical flags are evaluated
        ts_df = app_stats.time_series_df
        assert 'Hydro_P_Value' in ts_df.columns, "P-value column missing for Hydrophobicity."

        # Test 2: Verify Residue lists are exported
        score_df = app_stats.scores_df
        assert 'Residue_List' in score_df.columns, "Residue lists were not exported properly."

        print("[+] Tests Passed: PCA projection, gridding, and T-tests operate successfully.")

        # Cleanup test files
        for f in os.listdir('.'):
            if f.startswith('test_merged'):
                os.remove(f)
    finally:
        os.remove(temp_pdb)
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDB Grid Pro - Evaluate region-based model variance.")
    parser.add_argument("pdb", help="Input multi-model PDB", nargs='?')
    parser.add_argument("--grid", type=float, default=20.0, help="Grid size in Angstroms")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for charge calculation")
    parser.add_argument("--ref_model", type=int, default=8, help="0-indexed reference model (Default: 8)")
    parser.add_argument("--weights", type=str, default="1.0,1.0,1.0",
                        help="Weights for Hydro,RMSD,Charge (Default: 1.0,1.0,1.0)")
    parser.add_argument("--sim_mode", action="store_true", help="Invert scoring so higher = greater similarity")
    parser.add_argument("--score_out", default="nav_analysis", help="Output file prefix for scoring")
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit")

    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if not args.pdb:
        parser.print_help()
        sys.exit(1)

    weights_tuple = tuple(map(float, args.weights.split(',')))

    app = PDBGridPro(args.pdb, grid_size=args.grid, ph=args.ph,
                     ref_model_idx=args.ref_model, weights=weights_tuple, sim_mode=args.sim_mode)
    app.run_pipeline(args.score_out)