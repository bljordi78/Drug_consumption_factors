
# PERSONALITY FACTORS THAT INFLUENCE DRUG CONSUMPTION #####################
# NATIONAL COLLEGE OF IRELAND 
# HIGHER DIPLOMA IN SCIENCE IN DATA ANALYTICS
# Project for Data AND web Mining
# by Jordi Batlle (x17133246)


# PART 1: TIDY DATA 
# _________________________________________________________________________

graphics.off()
rm(list=ls())
ls(all.names=TRUE)


# Load libraries
library(plyr) 
library(dplyr) 
library(factoextra) 
library("caret") 
library(partykit) 
library(rpart) 
library(randomForest) 


# Import data frame from UCI repository
df <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data", header = FALSE, sep = ",")

# Research focus only on Ecstasy
# Remove other drugs data
df = df[,c(2:13,23)]

# Set column names
colnames(df) <- c("Age","Gender","Education","Country","Ethnicity","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness","Impulsiveness","Sensation","Ecstasy") 

# Check dataset details
head(df)
glimpse(df)
summary(df)


# As the amount of records is low and there too many classes 
# Aggregate and make a 3-classes research
df[,13]=revalue(df[,13], c(
  "CL0"=0,  # Never Used
  "CL1"=1,  # Used over a Decade Ago
  "CL2"=1,  # Used in Last Decade
  "CL3"=1,  # Used in Last Year
  "CL4"=2,  # Used in Last Month
  "CL5"=2,  # Used in Last Week
  "CL6"=2)) # Used in Last Day


# Check how data is balanced after revaluation
summary(df[,13])

# Check missing values
anyNA(df)


# Undersampling to balance the classes
# 1.Create a dataframe for each class
set.seed(123)
df0 <- df[which(df[,13] == "0"),]
df1 <- df[which(df[,13] == "1"),]
df2 <- df[which(df[,13] == "2"),]

# 2.Find which dataset has the least rows
nrow(df0)
nrow(df1)
nrow(df2)

# 3. df2 has the least number of records: take a random sample of that number of rows from the other 2 data sets
df0 <- sample_n(df0, nrow(df2))
df1 <- sample_n(df1, nrow(df2))
df2 <- sample_n(df2, nrow(df2))

# 4. Combine the 3 datasets
df_comb <- rbind(df0,df1,df2)

# 5. Resample the dataset
df <- sample_n(df_comb, nrow(df_comb))
rm(df_comb, df0, df1, df2)
dim(df)


# Due to undersampling, many records were lost to balance the classes
# As number of records is low, will use cross validation to increase the training stage


# PART 2: DATA EXPLORATION
# _________________________________________________________________________

# Pairs vs Correlation Matrix: ============================================

# Correlation panel
my_cols <- c("blue", "green3", "red" )
panel.cor<-function(x, y){
  usr <-par("usr"); 
  on.exit(par(usr))
  par(usr=c(0, 1, 0, 1))
  r <-round(cor(x, y), digits=2)
  txt <-paste0("", r)
  text(0.5, 0.5, txt, cex = 6* abs(cor(x, y)))}
# Customize upper panel
panel.pairs <-function(x, y){
  points(x,y, pch = 1, col = my_cols[df[,13]])}
# Create the plots
pairs(df, upper.panel = panel.pairs, lower.panel = panel.cor)

# Delete higher multicolinear variables: "Impulsiveness" vs "Sensation" r=0,62
df = df[,c(-12)]


# Boxplots do not show very extreme values
boxplot(df[-13], 
        main=paste("Boxplots of","all variables",sep=" "),
        xlab="Variables", ylab="Values", col=rainbow(7), pch="*",las=1)


# Check final dataset details
head(df)
glimpse(df)
summary(df)



# PART 3: MACHINE LEARNING ALGORITHMS    ##################################
# K-Means / Classification Tree / KNN / Logistic Regression / Kernel SVM / Naive Bayes
#______________________________________________________________________________________


## K-Means: check if it is possible to define clusters of data ########################

# Set max number of clusters
k.max <- 20

# Scale the data to standardise values range
df.scaled <- scale(df[-12])

wss <- sapply(1:k.max, function(k){
  kmeans(df.scaled, k, nstart=1 )$tot.withinss})

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# Elbow can be found for k=15
K=15
abline(v=K, lty=2)

# Run K-Means
km.res <- kmeans(df.scaled, K, nstart = 25)

fviz_cluster(km.res, 
             data=df.scaled, 
             geom="point",
             stand=FALSE, 
             show.clust.cent=TRUE,
             pointsize=1,
             ellipse = TRUE,
             ellipse.type = "convex",
             main = "DRUGS DATASET CLUSTERS",
             outlier.color="black")

# K-MEANS did not show the data could be split into clusters


# Split data into training and test to apply to classification algorithms
# 75% train / 25% test
set.seed(123)
N <- nrow(df)
traindf <- sample(1:N,size=0.75*N)
traindf <- sort(traindf)
testdf <- setdiff(1:N,traindf)


# Run a k-Nearest Neighbors KNN
tc <- trainControl(method="repeatedcv",number=10, repeats=3)
fitknn <- train(Ecstasy~., data=df[traindf,], method = "knn", trControl= tc, preProcess= c("center", "scale"), tuneLength= 10)
fitknn
# Plot accuracy vs K Value graph
plot(fitknn)


# Run a Classification Tree on the data:
fittree <- train(Ecstasy~., data = df[traindf,], method = "rpart", trControl = tc,  tuneLength = 10)
fittree
plot(fittree$finalModel)
text(fittree$finalModel, digits = 3)


# Run a Random Forest on the data:
fitrf <- randomForest(Ecstasy~., data=df[traindf,], ntree=150, importance=TRUE)
fitrf
(varimp <- importance(fitrf))
varImpPlot(fitrf, n.var=min(10, nrow(fitrf$importance)), main="TOP 10 VARIABLE IN RANDOM FOREST")


# Use the models to Predict classes on the test data
predknn <- predict(fitknn, newdata=df, subset=testdf)
predtree <- predict(fittree, newdata=df, subset=testdf)
predrf <- predict(fitrf, newdata=df, subset=testdf)


# Confusion matrices for test results
confusionMatrix(df[testdf, 12], predknn[testdf])
confusionMatrix(df[testdf, 12], predtree[testdf])
confusionMatrix(df[testdf, 12], predrf[testdf])
