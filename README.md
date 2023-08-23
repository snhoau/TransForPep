# TransForPep

This program uses a trained Transformer model[arXiv:1706.03762] to generate atomic coordinates, types, and charges for protein interaction surfaces on any given protein surface region. The training data for this model is extracted from the PDB database. Based on this data, the program can use the Point Drift algorithm alignment [DOI:10.1109/TPAMI.2010.46] method to search for peptides in the database (which should have a stable secondary structure and calculated atomic charges, and can be stable peptides or cyclic peptides) to obtain backbone proteins. Finally, software like Rosetta is used for side-chain optimization of the interaction site to obtain interacting peptides at specified protein residues.

The training dataset can be constructed by the user or can be obtained from a pre-built database (https://u.pcloud.link/publink/show?code=XZFQeaVZlRnVZny3nygSc7h4j5llMY1NW6SUFoVXy).

1_transformer_redo_ignh_dis350_cbem.py: Program for training the model.

2_transformer_evel_dis350_cbem.py: Program that calls the trained model for outputting the interaction interfaces.

3_ICPdatabase_cbem.py: Program that uses the alignment method to search the interaction interfaces in the database.

4_score.py: Program that calls the results of the alignment and scores the peptides in the database.

supplymentary/cbem.pkl: Contains a combination dictionary of atomic types and charges.

supplymentary/transpep_model_dis350_cbem_final.pt: Trained model file.

============================================================================================

这个程序是使用从PDB数据库中提取出来的蛋白质互作界面训练Transformer (arXiv:1706.03762)模型，使其可以生成蛋白任意表面区域的互作表面的原子坐标、类型和电荷，以此数据为基础可以使用Point Drift algorithm配准方法(DOI:10.1109/TPAMI.2010.46)对数据库中的多肽（需结构本身较稳定，并计算了原子电荷，可以是有稳定二级结构的多肽或者环肽）进行搜索以获得骨架蛋白，最后使用Rosetta等蛋白互作优化软件进行互作位的侧链优化，以期获得指定蛋白位点的互作多肽。

训练数据集，可以自己构建，也可使用我构建好的数据库

1_transformer_redo_ignh_dis350_cbem.py 训练模型的程序

2_transformer_evel_dis350_cbem.py 调用训练好的模型，用于互作界面的输出

3_ICPdatabase_cbem.py 使用配准方法对互作界面进行数据库搜索

4_score.py 调用上一步配准的结果，并对数据库中的多肽进行打分

supplymentary/cbem.pkl 包含原子类型和电荷的组合字典

supplymentary/transpep_model_dis350_cbem_final.pt 为训练好的模型
