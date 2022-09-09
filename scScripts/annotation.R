#install.packages("reticulate")
library(reticulate)
#py_install("anndata")
library(anndata)
library(dplyr)
fastAUC <- function(probs, class) {
  x <- probs
  y <- class
  x1 = x[y==1]; n1 = length(x1); 
  x2 = x[y==0]; n2 = length(x2);
  r = rank(c(x1,x2))  
  auc = (sum(r[1:n1]) - n1*(n1+1)/2) / n1 / n2
  return(auc)
}  


library(dplyr)
library(foreach)
library(doParallel)
library(Matrix)

saveRDSlist = c("'D:/COVID5/wop4-anno-ni.rds")
saveh5adlist = c("D:/COVID5/nCoV-wop4.h5ad")
savejpeglist = c("D:/COVID5/wop4-anno.pdf")
savejpeglist2 = c("D:/COVID5/wop4-anno2.pdf")
saveAUClist = c("D:/COVID5/wop4-auc.csv")
savescorelist = c("D:/COVID5/wop4-score.csv")
saveRDSlist2 = c("D:/COVID5/wop4-anno-allgenes.rds")

for(f in 1){
  data_ad <- read_h5ad(saveh5adlist) # african4d2
  #data_df <- as.data.frame(data_ad@assays$RNA@counts)
  #expr_matrix <- expr_matrix[tabulate(summary(expr_matrix)$i) != 0, , drop = FALSE]
  #expr_matrix <- as.matrix(expr_matrix)
  
  data_df <- data_ad$to_df()
  cmatch <- read.csv("D:/COVID5/cmatch-lung.csv")
  ###brain filter
  cmatch <-filter(cmatch,cancer=="Normal")
  ######
  cell_names <- unique(cmatch[,"celltype"])
  data_df <- 2^(data_df)
  temp_data <- data_df
  empty_df <- data.frame(matrix(NA, nrow = nrow(temp_data), ncol = length(cell_names)))
  col_index=0
  
  print(Sys.time())
  
  #### SCORE CALCULATION ####
  
  for(i in 1:length(cell_names)){
    print(i)
    oligo <- filter(cmatch, cmatch$celltype == cell_names[i])
    hits<- intersect(names(data_df),oligo$gene)
    col_index = list()
    for(j in 1:length(hits)){
      col_index = append(col_index,which(names(data_df)==hits[j])) #column indexes of hits
    }
    row_sum = apply(temp_data[,unlist(col_index),drop=F],1,sum) #sum of marker RNA hits in cell
    normalization <- apply((temp_data),1,sum) #sum of all RNA hits in cell
    empty_df[,i]<- log10((((row_sum/normalization)*10000)+1))
    #normalization of score 
    #this makes the function very slow
    #can take natural log and divide by 2.3
  }
  
  colnames(empty_df) <- cell_names
  empty_df$clusters <- data_ad$obs$seurat_clusters+1
  max(data_ad$obs$seurat_clusters)
  empty_df[is.na(empty_df)] = 0
  print(Sys.time())
  write.csv(empty_df,savescorelist[f])
  ############################
  print("enter auc")
  
  
  #### AUC CALCULATION #######
  fullAUCcalc <- function(empty_df,duplicates){
    clusterrange = max(empty_df$clusters)
    auc_df <- data.frame(matrix(NA, nrow = clusterrange, 
                                ncol = length(cell_names)))
    #print(clusterrange)
    library(Metrics)
    #print(Sys.time())
    for(i in 1:clusterrange){
      true = filter(empty_df,clusters==i)
      false = filter(empty_df,clusters!=i)
      aucs=list()
      #print(i)
      for(j in 1:length(cell_names)){
        library(Metrics)
        min = pmin(nrow(true),nrow(false))
        
        predicted<-append(true[sample(nrow(true),min),j],
                          false[sample(nrow(true),min),j])
        label <- 1:(2*min)
        for(k in 1:min){
          label[k] = 1
        }
        for(k in min:(2*min)){
          label[k] = 0
        }
        auc_val<-fastAUC(predicted,label)
        ### THIS IS 10x FASTER THAN ROCR ###
        #print(auc_val)
        aucs <- append(aucs,auc_val)
      }
      #print(aucs)
      auc_df[i,] <- aucs
    }
    return(auc_df)
  }
  
  ### AUC DUP CALCULATION ####
  fullAUCcalc_duplicates <- function(empty_df,duplicates){
    clusterrange = max(empty_df$clusters)
    auc_df <- data.frame(matrix(NA, nrow = clusterrange, 
                                ncol = length(cell_names)))
    library(Metrics)
    for(i in 1:clusterrange){
      true = filter(empty_df,clusters==i)
      '%ni%' <- Negate('%in%')
      false = filter(empty_df,clusters!=i & clusters%ni%duplicates)
      aucs=list()
      #print(i)
      for(j in 1:length(cell_names)){
        library(Metrics)
        min = pmin(nrow(true),nrow(false))
        
        predicted<-append(true[sample(nrow(true),min),j],
                          false[sample(nrow(true),min),j])
        label <- 1:(2*min)
        for(k in 1:min){
          label[k] = 1
        }
        for(k in min:(2*min)){
          label[k] = 0
        }
        auc_val<-fastAUC(predicted,label)
        ### THIS IS 10x FASTER THAN ROCR ###
        #print(auc_val)
        aucs <- append(aucs,auc_val)
      }
      auc_df[i,] <- aucs
    }
    return(auc_df)
  }
  ############################
  print(Sys.time())
  #print(dim(auc_df))
  
  ###After initial assignment of celltypes to clusters, we recomputed the AUC of a cluster for a cell
  #type by excluding other clusters of cells that were assigned to that cell type. This process was #repeated until there we no changes in the cluster assignment. WE then calculated the AUC for a cell
  #type by merging the cluster of cells that were assigned to that cell type.
  ###Generate AUC Scores#
  auc_df<-fullAUCcalc(empty_df)
  colnames(auc_df)<-cell_names
  ##Save original AUC ###
  write.csv(auc_df,saveAUClist[f])
  #######################
  cellname_df <- data.frame(matrix(data=NA,ncol=3,nrow=(nrow(auc_df)-2)))
  for (i in 1:nrow(auc_df)){
    cellname_index <-which.max(auc_df[i,])
    cellname_df[i,1]<- i-1
    cellname_df[i,2]<-cell_names[cellname_index]
    cellname_df[i,3]<- max(auc_df[i,])
  }
  #### Generate original Annotated Clusters ####
  colnames(cellname_df)<-c("Cluster_Number","Cell_Type","AUC Score")
  print(cellname_df)
  print(dim(cellname_df))
  ##############################################
  
  ####Cluster Reassignment Function###
  cluster_names = list()
  renameclusters<- function(auc_df){
    cluster_names = data.frame(matrix(data=NA,ncol=3,nrow=nrow(auc_df)))
    colnames(cluster_names)<-c("Cluster","Cell_Name","AUC_Score")
    for(i in 1:nrow(auc_df)){
      cluster_names$Cluster[i]<-i
      duplicates = list()
      #check for duplicates and get clusterID
      for(j in 1:nrow(auc_df)){
        if(which.max(auc_df[j,]) == which.max(auc_df[i,])){
          duplicates <- c(duplicates,j)
        }
      }
      
      if(length(duplicates)>1){
        newAUC<-fullAUCcalc_duplicates(empty_df,duplicates)
        cluster_names$Cell_Name[i]<-cell_names[which.max(newAUC[i,])]
        cluster_names$AUC_Score[i]<-max(newAUC[i,])
        #print(max(newAUC[21,]))
        #print("dup>1 added")
      }
      else{
        cluster_names$Cell_Name[i]<-cell_names[which.max(auc_df[i,])]
        cluster_names$AUC_Score[i]<-max(auc_df[i,])
        #print("nodup added")
      }
    }
    return(cluster_names)
  }
  #####################################
  #cluster_names <- renameclusters(auc_df)
  
  library(Seurat)
  library(plyr)
  library(SeuratObject)
  library(SeuratDisk)
  seurat<-LoadH5Seurat("D:/COVID5/nCoV-wop4.h5seurat")
  ########## Save Original Cluster Assignments###
  #test_cell_names <- as.list(cellname_df$Cell_Type)
  test_cell_names <- cellname_df$Cell_Type
  #names(test_cell_names )<-levels(seurat)
  seurat@meta.data$celltype <- seurat@meta.data$seurat_clusters
  seurat@meta.data$celltype <- mapvalues(seurat@meta.data$celltype,from = (cellname_df$Cluster_Number),
                                         to=(cellname_df$Cell_Type))
  
  
  #seurat<-RenameIdents(seurat,test_cell_names )
  pdf(savejpeglist[f])
  print(DimPlot(seurat,label=TRUE, group.by = "celltype",reduction = "umap",label.size = 3))
  dev.off
  DimPlot(seurat,group.by = "orig.ident",reduction="umap")
  saveRDS(seurat,file = saveRDSlist2[f])
  SaveH5Seurat(seurat, "D:/COVID5/nCoV-wop4-ct.h5seurat")
  Convert(source ='D:/COVID5/nCoV-wop4-ct.h5seurat',dest= 'h5ad',assay = 'SCT') 
  a<-seurat$celltype
  write.csv(a,file="D:/COVID5/wop4-ni-ct.csv")
  print("donezo")
  ###############################################
}
test_cell_names
seurat@meta.data

