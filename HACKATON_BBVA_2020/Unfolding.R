#Victor Manuel Gómez Espinosa
#librerias
library("smacof")
library("readxl")

#se leen los datos de similaridades o disimilaridades

preferencias <- read_excel("preferencias.xlsx") #disimilaridades

#nombres bancos
nombres<- names(preferencias)

#dim(preferencias)[1]

#ratings <- sim2diss(disim_nations, method = 15)  ## reverse ratings
ratings <- preferencias
rownames(ratings) <- c(1:dim(preferencias)[1]) #1 hasta el numero de filas


unf_ord <- unfolding(ratings, type = "ordinal")
#unf_ord

startconf <- list(unf_ord$conf.row, unf_ord$conf.col)
unf_cond <- unfolding(ratings, circle="column",type = "ordinal", conditionality = "row", eps = 6e-5, init = startconf)
#unf_cond

#unf_cond$conf.row

#unf_cond$conf.col
#plot(unf_cond, plot.type = "Shepard")


#plot(unf_cond)





write.csv(unf_cond$conf.row,file = "rows.csv")
write.csv(unf_cond$conf.col,file = "cols.csv")






