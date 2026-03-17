# AdaptIPs
AdaptIPs is an adaptive deep learning model built on dual-channel feature fusion and transfer learning. It is specifically designed for the high-throughput and high-accuracy prediction of protein phosphorylation sites, with a particular focus on addressing the challenge of phosphorylation site identification in low-resource and small-sample settings in bioinformatics. It provides efficient computational support for studies of protein modification, disease mechanism analysis, and drug development.

Abstract

Protein phosphorylation, a core post-translational modification of proteins, is deeply involved in pivotal biological activities including cell signal transduction and disease pathogenesis. Accordingly, the accurate and efficient prediction of phosphorylation sites is critical for elucidating molecular mechanisms of diseases and advancing the development of targeted therapies. However, existing prediction methods for phosphorylation sites still face several intractable bottlenecks, such as insufficient feature extraction, limited model generalization ability, and poor recognition accuracy and stability, especially for phosphorylation sites with scarce sample resources.
To address these pressing challenges, this study proposes AdaptIPs, a novel dual-channel fusion and adaptive deep learning framework dedicated to the precise identification of protein phosphorylation sites. This framework innovatively adopts a dual-path parallel architecture to synchronously extract core local sequence features and global semantic deep representations derived from large-scale biological pre-trained language models. It further integrates the dual-path features efficiently via feature multiplication fusion, coupled with an adaptive attention mechanism to enhance the weight of key features, and adopts an adaptive early stopping strategy to prevent model overfitting, thereby achieving efficient and robust identification of phosphorylation sites.
To tackle the critical problem of data scarcity for tyrosine (Y) phosphorylation sites, a typical low-resource biological data scenario, this study further introduces a transfer learning strategy. Specifically, the basic model is pre-trained on sufficient data of serine/threonine (S/T) phosphorylation sites, followed by fine-tuning to realize cross-site knowledge transfer. This approach effectively reuses mature modeling experience and feature representation capabilities, drastically improving the prediction performance of low-resource Y phosphorylation sites.
Independent dataset validation demonstrates the outstanding predictive performance of AdaptIPs: for S/T phosphorylation sites, the identification accuracy reaches 84.48% and the AUC value is 91.83%; for scarce Y phosphorylation sites, the prediction accuracy and AUC value rise to 92.86% and 94.33% respectively. The model outperforms existing mainstream phosphorylation site prediction methods on multiple key evaluation metrics, exhibiting exceptional comprehensive performance and cross-scenario generalization ability.
The AdaptIPs framework developed in this study not only provides a powerful and reliable computational tool for high-throughput and high-precision prediction of phosphorylation sites, but also offers a novel paradigm for solving few-shot learning problems in other biological sequence analyses via its proposed dual-channel feature fusion combined with transfer learning. It lays a more solid computational and theoretical foundation for the basic research and clinical translation of related diseases.









Requirement

python                    3.10.0
pytorch                   2.2.0  
numpy                     1.24.3  
NVIDIA A100GPU
