# MGATE
Motivation: Predicting disease-related long non-coding RNAs (lncRNAs) can be used as the biomarkers for disease diagnosis and
treatment. The development of effective computational prediction approaches to predict lncRNA-disease associations (LDAs) can
provide insights into the pathogenesis of complex human diseases and reduce experimental costs. However, few of the existing
methods use microRNA (miRNA) information and consider the complex relationship between inter-graph and intra-graph in complex-
graph for assisting prediction.

Results: In this paper, the relationships between the same types of nodes and different types of nodes in complex-graph are
introduced. We propose a multi-channel graph attention autoencoder model to predict LDAs, called MGATE. First, an lncRNA-miRNA-
disease complex-graph is established based on the similarity and correlation among lncRNA, miRNA and diseases to integrate
the complex association among them. Secondly, in order to fully extract the comprehensive information of the nodes, we use
graph autoencoder networks to learn multiple representations from complex-graph, inter-graph and intra-graph. Thirdly, a graph-
level attention mechanism integration module is adopted to adaptively merge the three representations, and a combined training
strategy is performed to optimize the whole model to ensure the complementary and consistency among the multi-graph embedding
representations. Finally, multiple classifiers are explored, and Random Forest is used to predict the association score between lncRNA
and disease. Experimental results on the public dataset show that the area under receiver operating characteristic curve and area
under precision-recall curve of MGATE are 0.964 and 0.413, respectively. MGATE performance significantly outperformed seven state-
of-the-art methods. Furthermore, the case studies of three cancers further demonstrate the ability of MGATE to identify potential
disease-correlated candidate lncRNAs


![image](https://user-images.githubusercontent.com/95516781/154301202-167f6fab-bb5c-4eb9-b1e3-27dc38005011.png)
