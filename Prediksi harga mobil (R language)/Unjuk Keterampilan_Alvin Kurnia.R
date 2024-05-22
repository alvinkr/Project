#load data
library(here)
df<- read.csv(here("mobil_mesin_harga.csv"))

#cekdata
summary(df)
plot(df)

#splidata
library(caTools)
set.seed(25)
df_split = sample.split(df$Harga,SplitRatio=0.8)
df_train= subset(df, df_split==TRUE)
df_test = subset(df, df_split==FALSE)

#membangun model
model=lm(formula = Harga ~ KekuatanMesin, data = df_train)
summary(model)

prediksi_harga=predict(model,newdata = df_test)
prediksi_harga

result = cbind(df_test,'hasil Prediksi'=prediksi_harga)
result

sqrt(mean(model$residuals^2))

library(ggforce)

ggplot()+geom_point(aes
                    (x=df_train$KekuatanMesin, y= df_train$Harga),colour= 'red')+
  geom_line(aes(x=df_train$KekuatanMesin, y=predict(model,newdata = df_train)),colour='blue')+
  ggtitle('Harga vs Kekuatan mesin (train)')+
  xlab('Kekuatan mesin')+ylab('Harga')

ggplot()+geom_point(aes
                    (x=df_test$KekuatanMesin, y= df_test$Harga),colour= 'red')+
  geom_line(aes(x=df_test$KekuatanMesin, y=predict(model,newdata = df_test)),colour='blue')+
  ggtitle('Harga vs Kekuatan Mesin (test)')+
  xlab('Kekuatan Mesin')+ylab('Harga')

hargacoba =data.frame(KekuatanMesin = c(200,180,160))
prediksiharga = predict(model, newdata = hargacoba)
prediksiharga

