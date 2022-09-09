library(Seurat)
library(dplyr)
library(Seurat)
library(patchwork)
library(SeuratDisk)
library(tidyverse)
ARNA <- readRDS(file = "D:/COVID5/nCoV.rds")
ARNA[['integrated']]<-NULL
#
#
ARNA[["percent.mt"]] <- PercentageFeatureSet(ARNA, pattern = "^MT-")
VlnPlot(ARNA, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

#C2 <- readRDS(file = "C:/Users/karla/Documents/Research COVID/integration-covid-scT.rds")

#A1 <- readRDS(file = "D:/COVID5/nCoV.rds")
#A1$sample_new<-A1$sample_new
#A1$group<-A1$group
#A$group1<-A1$group1
A1<-ARNA
A <- PercentageFeatureSet(A1, pattern = "^MT-", col.name = "percent.mt") # get the percent.mt
#A1 <- subset(A, subset = sample_new != "HC4")
A1 <- subset(A1, subset = sample_new != "C1")
A1 <- subset(A1, subset = sample_new != "C2")
A1 <- subset(A1, subset = sample_new != "C3")
A1 <- subset(A1, subset = sample_new != "O1")
A1 <- subset(A1, subset = sample_new != "O2")
A1 <- subset(A1, subset = sample_new != "O3")
A1 <- subset(A1, subset = sample_new != "HC1")
A1 <- subset(A1, subset = sample_new != "HC2")
A1 <- subset(A1, subset = sample_new != "HC3")
A1<-UpdateSeuratObject(A1)
A1<-DietSeurat(
  A1,
  counts = TRUE,
  data = TRUE,
  scale.data = FALSE,
  features = NULL,
  assays = 'RNA',
)

 

VlnPlot(A1,features = c("nFeature_RNA", "nCount_RNA","percent.mt"))
A <- SCTransform(A1, vars.to.regress = "percent.mt", verbose = FALSE) 
A <- RunPCA(A, verbose = FALSE)
A <- RunUMAP(A, dims = 1:30, verbose = FALSE)
A <- FindNeighbors(A, dims = 1:30, verbose = FALSE)
A <- FindClusters(A, verbose = FALSE, resolution = 0.32)
DimPlot(A, label = TRUE)
DimPlot(object = A, split.by = 'sample_new')
DimPlot(A, reduction = "umap", group.by = "sample_new")
SaveH5Seurat(A, "D:/COVID5/nCoV-wop4.h5seurat")
Convert(source ='D:/COVID5/nCoV-wop4.h5seurat',dest= 'h5ad',assay = 'SCT') 
#saveRDS(A,"C:/Users/karla/Documents/Research COVID/imputation-covid-scT.rds") # save sctransform result


A <- LoadH5Seurat("D:/COVID5/nCoV-wop4.h5seurat")
#Integration
ifnb.list <- SplitObject(A, split.by = "sample_new")
#ifnb.list <- lapply(X = ifnb.list, FUN = function(x) {
  #x <- NormalizeData(x, normalization.method='RC')
#  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
#})
features <- SelectIntegrationFeatures(object.list = ifnb.list)
immune.anchors <- FindIntegrationAnchors(object.list = ifnb.list, anchor.features = features, normalization.method = "LogNormalize")
immune.combined1<-IntegrateData(anchorset = immune.anchors, normalization.method = "LogNormalize")
#immune.combined <- IntegrateData(anchorset = immune.anchors, features.to.integrate = features,normalization.method = "SCT",)
DefaultAssay(immune.combined1) <- "integrated"
immune.combined1 <- ScaleData(immune.combined1, verbose = TRUE,do.center = FALSE)
immune.combined1 <- RunPCA(immune.combined1, npcs = 30, verbose = TRUE)
immune.combined1 <- RunUMAP(immune.combined1, reduction = "pca", dims = 1:30)
immune.combined1 <- FindNeighbors(immune.combined1, reduction = "pca", dims = 1:30)
immune.combined1 <- FindClusters(immune.combined1, resolution = 1.21)

p1 <- DimPlot(immune.combined1, reduction = "umap", group.by = "sample_new")
p2 <- DimPlot(immune.combined1, reduction = "umap", label = TRUE, repel = TRUE)
p1 + p2

SaveH5Seurat(immune.combined1, "D:/COVID5/nCoV-wop4.h5seurat")
Convert(source ='D:/COVID5/nCoV-wop4.h5seurat',dest= 'h5ad',assay = 'integrated') 

