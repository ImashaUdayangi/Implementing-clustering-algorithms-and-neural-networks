#Load the libraries 
library(readxl)
library(NbClust)
library(factoextra)
library(cluster)
library(fpc)

#Reading the WhiteWineFile
WhitewineData <- read_xlsx("Whitewine_v6.xlsx")

#Part a
##################################### Pre-proccesing ##############################################

# Subset the dataset to include only the first 11 attributes
First11Attributes_data <- WhitewineData[, 1:11]

# Check for missing data
sum(is.na(First11Attributes_data))

# show the box plot with outliers
boxplot(First11Attributes_data)

# Outlier Removal
outlierRemoval = apply(First11Attributes_data, 2, function(x) {
  median_value = median(x)
  mad = median(abs(x - median_value))
  return(x < (median_value - 3 * mad) | x > (median_value + 3 * mad))
})

First11Attributes_data_without_outliers = First11Attributes_data[!apply(outlierRemoval, 1, any),]

# show the box plot without outliers
boxplot(First11Attributes_data_without_outliers)

# Normalization
First11Attributes_data_scaled = scale(First11Attributes_data_without_outliers)
#show the box plot
boxplot(First11Attributes_data_scaled )


#Part b
# Compute the optimal number of clusters using NbClust Method
NbClust(First11Attributes_data_scaled, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans", index="all")

# Compute the gap statistic for each value of 
set.seed(123)
gap_stat <- clusGap(First11Attributes_data_scaled, FUN = kmeans, nstart = 25, K.max = 10, B = 50)

fviz_gap_stat(gap_stat)

# Compute the optimal number of clusters using Elbow Method
fviz_nbclust(First11Attributes_data_scaled, kmeans, method = "wss") + geom_vline(xintercept = 2, linetype = 2) + labs(title = "Elbow Method")

# Compute the optimal number of clusters using Silhouette Method
fviz_nbclust(First11Attributes_data_scaled, kmeans, method = "silhouette")

#Part c
# Define the number of clusters to be formed
k = 2

# Perform k-means clustering on the scaled wine dataset using the specified number of clusters and 10 random starts
kmeans_Data2 <- kmeans(First11Attributes_data_scaled, centers = k, nstart = 10)

# Print the k-means clustering results, including the cluster centers and assignments
kmeans_Data2

# Visualize the clustering results using the factoextra package, which creates a scatter plot with the observations colored according to their cluster assignments
fviz_cluster(kmeans_Data2, First11Attributes_data_scaled, palette = c("#ff7f00","#836fff"), geom = "point", ellipse.type = "convex", ggtheme = theme_bw())

# Extract the total within-cluster sum of squares (WSS), which is a measure of the compactness of the clusters
wss <- kmeans_Data2$tot.withinss
wss

# Extract the between-cluster sum of squares (BSS), which is a measure of the separation between the clusters
bss <- kmeans_Data2$betweenss
bss

# Compute the total sum of squares (TSS), which is the sum of the WSS and BSS
tss <- kmeans_Data2$totss
tss

# Compute the ratio of the BSS to the TSS, which is a measure of the proportion of the total variance explained by the clustering
ratio <- (bss/tss)*100
ratio

#Part d
# Compute the silhouette widths for each observation,Applying silhouette for K=2
sil <- silhouette(kmeans_Data2$cluster, dist(First11Attributes_data_scaled))
fviz_silhouette(sil)



##################################### PCA ##############################################
#Part e
#View the scaled wine dataset
View(First11Attributes_data_scaled)

#Perform principal component analysis (PCA) on the scaled wine dataset using the prcomp function Center and scale the dataset as well
pcaData <- prcomp(First11Attributes_data_scaled, center = TRUE, scale = TRUE)
summary(pcaData)

#Extract the eigenvalues and eigenvectors from the PCA results and view
eigenvalues <- pcaData$sdev^2
eigenvector <- pcaData$rotation
eigenvalues
eigenvector

#Calculate the cumulative score of the eigenvalues
cumulativeScore <- cumsum(eigenvalues/sum(eigenvalues))

#Select the eigenvalues whose cumulative score is less than or equal to 0.85
selectedValues <- which(cumulativeScore <= 0.85)
summary(selectedValues)

#Transform the original dataset using the selected eigenvectors
transformDataset <- as.data.frame(pcaData$x[,selectedValues])
summary(transformDataset)


#Part f
# Determine the optimal number of clusters using the NbClust method for PCA
nb_pca <- NbClust(transformDataset, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Determine the optimal number of clusters using the elbow method for PCA
fviz_nbclust(transformDataset, kmeans, method = "wss")+ geom_vline(xintercept = 2, linetype = 2) + labs(title = "Elbow Method")

# Compute the optimal number of clusters using Silhouette Method for PCA
fviz_nbclust(transformDataset, kmeans, method = "silhouette")

# Determine the optimal number of clusters using the gap statistic method for PCA
gap_stat <- clusGap(transformDataset, kmeans, K.max = 10, nstart = 10, B=50)
fviz_gap_stat(gap_stat)

#Part g
# Set the number of clusters based on the most favored k value
k <- 2

# Perform k-means analysis on the transformed PCA dataset
kmeans_pca_data <- kmeans(transformDataset, centers = k, nstart = 10)

# Show the related kmeans output
kmeans_pca_data

# Create a scatter plot of the clustering results
fviz_cluster(kmeans_pca_data, transformDataset, palette = c("#ff7f00","#836fff"), geom = "point", ellipse.type = "convex", ggtheme = theme_bw())

# Extract the total within-cluster sum of squares (WSS), which is a measure of the compactness of the clusters
wss<- kmeans_pca_data$tot.withinss
wss

# Extract the between-cluster sum of squares (BSS), which is a measure of the separation between the clusters
bss <- kmeans_pca_data$betweenss
bss

# Compute the total sum of squares (TSS), which is the sum of the WSS and BSS
tss <- kmeans_pca_data$totss
tss

# Compute the ratio of the BSS to the TSS, which is a measure of the proportion of the total variance explained by the clustering
ratio <- (bss/tss)*100
ratio

#Part h
# Compute the silhouette widths for each observation,Applying silhouette for K=2
sil_pca <- silhouette(kmeans_pca_data$cluster, dist(transformDataset))
fviz_silhouette(sil_pca)

#Part i
#calculate the Calinski-Harabaz index k=2
ch_index <- calinhara(transformDataset, kmeans_pca_data$cluster)
print(ch_index)
