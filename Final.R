rm(list=ls())
library(data.table)
library(lubridate)
library(xgboost)
library(forecast)
library(ggplot2)

# Note : 
# No alarms
# No state Info
# No instal Info

# read train and test data.
train <- data.table(read.csv("train.csv"))
test <- data.table(read.csv("test.csv"))
# set date to POSIXct format
train[,Time:=ymd_hms(Time)]
test[,Time:=ymd_hms(Time)]
ds <- rbind(train,test)

# keep track of where the test set starts
testsetdates <- ds[is.na(Temperature),Time][1]

# read measures data
measures <- data.table(read.csv("../Data/measures.csv"))
measures[,Time:=ymd_hms(Time)]

######################################################################
##  Data Cleaning
######################################################################

## Cleaning for 2341753007 and 2341758001
# some outliers in Temperature. Replace by NA, will be interpolated later.
ds[NumInstallation == 2341753007&Temperature<7,Temperature:=NA]
ds[NumInstallation %in% c(2341753007,2341758001),DT:=(abs(Temperature-shift(Temperature,2,type='lag'))>1.4)]
ds[NumInstallation %in% c(2341753007,2341758001),DT2:=(abs(Temperature-shift(Temperature,2,type='lead'))>1.4)]
ds[NumInstallation %in% c(2341753007,2341758001),DT3:=(DT2*DT)]
# Note that one point is removed while being ok, but interpolatino
# will fixes it reasonably well so don't bother...
ds[NumInstallation %in% c(2341753007,2341758001)&DT3==1,Temperature:=NA]
ds[,':='(DT=NULL,
         DT2=NULL,
         DT3=NULL)]

## Variables to keep in the measures :
VarToKeep <- c("BASE_VN_T.EXTER_T.REF_CORRIGE",
               "CIR1_AI_EAU_T.DEP_SONDE","CIR1_AI_EAU_T.RET_SONDE",
               "CIR2_AI_EAU_T.DEP_SONDE","CIR2_AI_EAU_T.RET_SONDE",
               "CIR3_AI_EAU_T.DEP_SONDE","CIR3_AI_EAU_T.RET_SONDE",
               "CIR4_AI_EAU_T.DEP_SONDE","CIR4_AI_EAU_T.RET_SONDE",
               "CIR5_AI_EAU_T.DEP_SONDE","CIR5_AI_EAU_T.RET_SONDE",
               "CIR6_AI_EAU_T.DEP_SONDE","CIR6_AI_EAU_T.RET_SONDE",               
               "PRIM_AI_EAU_T.DEP_SONDE","PRIM_AI_EAU_T.RET_SONDE")
 
measures <- measures[Label %in% VarToKeep]
measures[,Label:=factor(Label)]
# Round each measures to the next hour
measures[,Time:=ceiling_date(Time,unit="hour")]

# Aggregate measyres by hour and installation number. Use mean temp as
# aggregate
xmeas <- dcast.data.table(Time+NumInstallation~Label,
                          value.var="Value",
                          fun.aggregate = mean,
                          data=measures)
# Correct Outlier
i <- which(xmeas$CIR1_AI_EAU_T.DEP_SONDE>500)
xmeas[i,':='(CIR1_AI_EAU_T.DEP_SONDE=NA,
             CIR1_AI_EAU_T.RET_SONDE=NA,
             CIR2_AI_EAU_T.DEP_SONDE=NA,
             CIR2_AI_EAU_T.RET_SONDE=NA)]

# merge measures and training/testing data 
ds <- merge(ds,xmeas,by=c("Time","NumInstallation"),all.x=TRUE)

# Function to correct Missing Values
ImputeMissing <- function(x){                                        
    if(sum(is.na(x))>0.9*(length(x)))x=NA  # if all values are NA - keep as NA
    else{                                  # if only one missing : interpolation
         x <- na.interp(x)
     }
    as.numeric(x)
}

# correct missing values for all columns of ds from the third.
Cols=names(ds)[3:length(names(ds))]
ds[,(Cols):=lapply(.SD,ImputeMissing),.SDcols=Cols,by=NumInstallation]

######################################################################
# Feature Engineering
######################################################################
## Reorder columns
setcolorder(ds,c(1,3,2,4:ncol(ds)))
## Reorder data set
ds <- ds[order(Time,NumInstallation)]
## Renames variables to ease coding
NewVarnames <- c("T_EXT",
               "C1_DEP","C1_RET",
               "C2_DEP","C2_RET",
               "C3_DEP","C3_RET",
               "C4_DEP","C4_RET",
               "C5_DEP","C5_RET",
               "C6_DEP","C6_RET",
               "P_DEP","P_RET")
setnames(ds,old=VarToKeep,new=NewVarnames)
## A bit more cleaning, these variables look strange. Best keep it out.
ds[NumInstallation==2310320005,C1_RET:=NA]
ds[NumInstallation==2310320005,C2_RET:=NA]
ds[C3_DEP==0,C3_DEP:=NA]
ds[C3_RET==0,C3_RET:=NA]
ds[NumInstallation==2310320005,C4_RET:=NA]
ds[NumInstallation==2310320005,C5_DEP:=NA]
ds[NumInstallation==2310320005,C5_RET:=NA]


## Create new variables as difference of between present measure and
## measures at 1 (then 2, 3, 4, ...10) hour before
ds[,':='(T_EXT_1=T_EXT-shift(T_EXT,1,type="lag"),
         C1_DEP_1=C1_DEP-shift(C1_DEP,1,type="lag"),
         C2_DEP_1=C2_DEP-shift(C2_DEP,1,type="lag"),
         C3_DEP_1=C3_DEP-shift(C3_DEP,1,type="lag"),
         C4_DEP_1=C4_DEP-shift(C4_DEP,1,type="lag"),
         C5_DEP_1=C5_DEP-shift(C5_DEP,1,type="lag"),
         C6_DEP_1=C6_DEP-shift(C6_DEP,1,type="lag"),
         C1_RET_1=C1_RET-shift(C1_RET,1,type="lag"),
         C2_RET_1=C2_RET-shift(C2_RET,1,type="lag"),
         C3_RET_1=C3_RET-shift(C3_RET,1,type="lag"),
         C4_RET_1=C4_RET-shift(C4_RET,1,type="lag"),
         C5_RET_1=C5_RET-shift(C5_RET,1,type="lag"),
         C6_RET_1=C6_RET-shift(C6_RET,1,type="lag"),
         P_DEP_1=P_DEP-shift(P_DEP,1,type="lag"),
         P_RET_1=P_RET-shift(P_RET,1,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_2=T_EXT-shift(T_EXT,2,type="lag"),
         C1_DEP_2=C1_DEP-shift(C1_DEP,2,type="lag"),
         C2_DEP_2=C2_DEP-shift(C2_DEP,2,type="lag"),
         C3_DEP_2=C3_DEP-shift(C3_DEP,2,type="lag"),
         C4_DEP_2=C4_DEP-shift(C4_DEP,2,type="lag"),
         C5_DEP_2=C5_DEP-shift(C5_DEP,2,type="lag"),
         C6_DEP_2=C6_DEP-shift(C6_DEP,2,type="lag"),
         C1_RET_2=C1_RET-shift(C1_RET,2,type="lag"),
         C2_RET_2=C2_RET-shift(C2_RET,2,type="lag"),
         C3_RET_2=C3_RET-shift(C3_RET,2,type="lag"),
         C4_RET_2=C4_RET-shift(C4_RET,2,type="lag"),
         C5_RET_2=C5_RET-shift(C5_RET,2,type="lag"),
         C6_RET_2=C6_RET-shift(C6_RET,2,type="lag"),
         P_DEP_2=P_DEP-shift(P_DEP,2,type="lag"),
         P_RET_2=P_RET-shift(P_RET,2,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_3=T_EXT-shift(T_EXT,3,type="lag"),
         C1_DEP_3=C1_DEP-shift(C1_DEP,3,type="lag"),
         C2_DEP_3=C2_DEP-shift(C2_DEP,3,type="lag"),
         C3_DEP_3=C3_DEP-shift(C3_DEP,3,type="lag"),
         C4_DEP_3=C4_DEP-shift(C4_DEP,3,type="lag"),
         C5_DEP_3=C5_DEP-shift(C5_DEP,3,type="lag"),
         C6_DEP_3=C6_DEP-shift(C6_DEP,3,type="lag"),
         C1_RET_3=C1_RET-shift(C1_RET,3,type="lag"),
         C2_RET_3=C2_RET-shift(C2_RET,3,type="lag"),
         C3_RET_3=C3_RET-shift(C3_RET,3,type="lag"),
         C4_RET_3=C4_RET-shift(C4_RET,3,type="lag"),
         C5_RET_3=C5_RET-shift(C5_RET,3,type="lag"),
         C6_RET_3=C6_RET-shift(C6_RET,3,type="lag"),
         P_DEP_3=P_DEP-shift(P_DEP,3,type="lag"),
         P_RET_3=P_RET-shift(P_RET,3,type="lag")
         ),by=NumInstallation]


ds[,':='(T_EXT_4=T_EXT-shift(T_EXT,4,type="lag"),
         C1_DEP_4=C1_DEP-shift(C1_DEP,4,type="lag"),
         C2_DEP_4=C2_DEP-shift(C2_DEP,4,type="lag"),
         C3_DEP_4=C3_DEP-shift(C3_DEP,4,type="lag"),
         C4_DEP_4=C4_DEP-shift(C4_DEP,4,type="lag"),
         C5_DEP_4=C5_DEP-shift(C5_DEP,4,type="lag"),
         C6_DEP_4=C6_DEP-shift(C6_DEP,4,type="lag"),
         C1_RET_4=C1_RET-shift(C1_RET,4,type="lag"),
         C2_RET_4=C2_RET-shift(C2_RET,4,type="lag"),
         C3_RET_4=C3_RET-shift(C3_RET,4,type="lag"),
         C4_RET_4=C4_RET-shift(C4_RET,4,type="lag"),
         C5_RET_4=C5_RET-shift(C5_RET,4,type="lag"),
         C6_RET_4=C6_RET-shift(C6_RET,4,type="lag"),
         P_DEP_4=P_DEP-shift(P_DEP,4,type="lag"),
         P_RET_4=P_RET-shift(P_RET,4,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_5=T_EXT-shift(T_EXT,5,type="lag"),
         C1_DEP_5=C1_DEP-shift(C1_DEP,5,type="lag"),
         C2_DEP_5=C2_DEP-shift(C2_DEP,5,type="lag"),
         C3_DEP_5=C3_DEP-shift(C3_DEP,5,type="lag"),
         C4_DEP_5=C4_DEP-shift(C4_DEP,5,type="lag"),
         C5_DEP_5=C5_DEP-shift(C5_DEP,5,type="lag"),
         C6_DEP_5=C6_DEP-shift(C6_DEP,5,type="lag"),
         C1_RET_5=C1_RET-shift(C1_RET,5,type="lag"),
         C2_RET_5=C2_RET-shift(C2_RET,5,type="lag"),
         C3_RET_5=C3_RET-shift(C3_RET,5,type="lag"),
         C4_RET_5=C4_RET-shift(C4_RET,5,type="lag"),
         C5_RET_5=C5_RET-shift(C5_RET,5,type="lag"),
         C6_RET_5=C6_RET-shift(C6_RET,5,type="lag"),
         P_DEP_5=P_DEP-shift(P_DEP,5,type="lag"),
         P_RET_5=P_RET-shift(P_RET,5,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_6=T_EXT-shift(T_EXT,6,type="lag"),
         C1_DEP_6=C1_DEP-shift(C1_DEP,6,type="lag"),
         C2_DEP_6=C2_DEP-shift(C2_DEP,6,type="lag"),
         C3_DEP_6=C3_DEP-shift(C3_DEP,6,type="lag"),
         C4_DEP_6=C4_DEP-shift(C4_DEP,6,type="lag"),
         C5_DEP_6=C5_DEP-shift(C5_DEP,6,type="lag"),
         C6_DEP_6=C6_DEP-shift(C6_DEP,6,type="lag"),
         C1_RET_6=C1_RET-shift(C1_RET,6,type="lag"),
         C2_RET_6=C2_RET-shift(C2_RET,6,type="lag"),
         C3_RET_6=C3_RET-shift(C3_RET,6,type="lag"),
         C4_RET_6=C4_RET-shift(C4_RET,6,type="lag"),
         C5_RET_6=C5_RET-shift(C5_RET,6,type="lag"),
         C6_RET_6=C6_RET-shift(C6_RET,6,type="lag"),
         P_DEP_6=P_DEP-shift(P_DEP,6,type="lag"),
         P_RET_6=P_RET-shift(P_RET,6,type="lag")
         ),by=NumInstallation]


ds[,':='(T_EXT_7=T_EXT-shift(T_EXT,7,type="lag"),
         C1_DEP_7=C1_DEP-shift(C1_DEP,7,type="lag"),
         C2_DEP_7=C2_DEP-shift(C2_DEP,7,type="lag"),
         C3_DEP_7=C3_DEP-shift(C3_DEP,7,type="lag"),
         C4_DEP_7=C4_DEP-shift(C4_DEP,7,type="lag"),
         C5_DEP_7=C5_DEP-shift(C5_DEP,7,type="lag"),
         C6_DEP_7=C6_DEP-shift(C6_DEP,7,type="lag"),
         C1_RET_7=C1_RET-shift(C1_RET,7,type="lag"),
         C2_RET_7=C2_RET-shift(C2_RET,7,type="lag"),
         C3_RET_7=C3_RET-shift(C3_RET,7,type="lag"),
         C4_RET_7=C4_RET-shift(C4_RET,7,type="lag"),
         C5_RET_7=C5_RET-shift(C5_RET,7,type="lag"),
         C6_RET_7=C6_RET-shift(C6_RET,7,type="lag"),
         P_DEP_7=P_DEP-shift(P_DEP,7,type="lag"),
         P_RET_7=P_RET-shift(P_RET,7,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_8=T_EXT-shift(T_EXT,8,type="lag"),
         C1_DEP_8=C1_DEP-shift(C1_DEP,8,type="lag"),
         C2_DEP_8=C2_DEP-shift(C2_DEP,8,type="lag"),
         C3_DEP_8=C3_DEP-shift(C3_DEP,8,type="lag"),
         C4_DEP_8=C4_DEP-shift(C4_DEP,8,type="lag"),
         C5_DEP_8=C5_DEP-shift(C5_DEP,8,type="lag"),
         C6_DEP_8=C6_DEP-shift(C6_DEP,8,type="lag"),
         C1_RET_8=C1_RET-shift(C1_RET,8,type="lag"),
         C2_RET_8=C2_RET-shift(C2_RET,8,type="lag"),
         C3_RET_8=C3_RET-shift(C3_RET,8,type="lag"),
         C4_RET_8=C4_RET-shift(C4_RET,8,type="lag"),
         C5_RET_8=C5_RET-shift(C5_RET,8,type="lag"),
         C6_RET_8=C6_RET-shift(C6_RET,8,type="lag"),
         P_DEP_8=P_DEP-shift(P_DEP,8,type="lag"),
         P_RET_8=P_RET-shift(P_RET,8,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_9=T_EXT-shift(T_EXT,9,type="lag"),
         C1_DEP_9=C1_DEP-shift(C1_DEP,9,type="lag"),
         C2_DEP_9=C2_DEP-shift(C2_DEP,9,type="lag"),
         C3_DEP_9=C3_DEP-shift(C3_DEP,9,type="lag"),
         C4_DEP_9=C4_DEP-shift(C4_DEP,9,type="lag"),
         C5_DEP_9=C5_DEP-shift(C5_DEP,9,type="lag"),
         C6_DEP_9=C6_DEP-shift(C6_DEP,9,type="lag"),
         C1_RET_9=C1_RET-shift(C1_RET,9,type="lag"),
         C2_RET_9=C2_RET-shift(C2_RET,9,type="lag"),
         C3_RET_9=C3_RET-shift(C3_RET,9,type="lag"),
         C4_RET_9=C4_RET-shift(C4_RET,9,type="lag"),
         C5_RET_9=C5_RET-shift(C5_RET,9,type="lag"),
         C6_RET_9=C6_RET-shift(C6_RET,9,type="lag"),
         P_DEP_9=P_DEP-shift(P_DEP,9,type="lag"),
         P_RET_9=P_RET-shift(P_RET,9,type="lag")
         ),by=NumInstallation]

ds[,':='(T_EXT_10=T_EXT-shift(T_EXT,10,type="lag"),
         C1_DEP_10=C1_DEP-shift(C1_DEP,10,type="lag"),
         C2_DEP_10=C2_DEP-shift(C2_DEP,10,type="lag"),
         C3_DEP_10=C3_DEP-shift(C3_DEP,10,type="lag"),
         C4_DEP_10=C4_DEP-shift(C4_DEP,10,type="lag"),
         C5_DEP_10=C5_DEP-shift(C5_DEP,10,type="lag"),
         C6_DEP_10=C6_DEP-shift(C6_DEP,10,type="lag"),
         C1_RET_10=C1_RET-shift(C1_RET,10,type="lag"),
         C2_RET_10=C2_RET-shift(C2_RET,10,type="lag"),
         C3_RET_10=C3_RET-shift(C3_RET,10,type="lag"),
         C4_RET_10=C4_RET-shift(C4_RET,10,type="lag"),
         C5_RET_10=C5_RET-shift(C5_RET,10,type="lag"),
         C6_RET_10=C6_RET-shift(C6_RET,10,type="lag"),
         P_DEP_10=P_DEP-shift(P_DEP,10,type="lag"),
         P_RET_10=P_RET-shift(P_RET,10,type="lag")
         ),by=NumInstallation]

# Add an indicator for schools, and another for elementary schools.
ds[,is_school:=ifelse(NumInstallation %in% c(9003489002,9003489001,9007420001),0,1)]
ds[,is_elem_school:=ifelse(NumInstallation %in% c(3324270001,3324270010),1,0)]

# reorder dataset (not necessarily needed, but it must remain in
# chronological order for the code to work.
ds <- ds[order(NumInstallation,Time)]
# add a time index
ds[,hId:=seq(1:.N),by=NumInstallation]
# order by installation number
ds <- ds[order(NumInstallation)]
# add an indicator for weekends and holidays
ds[,is_holiday:=ifelse(weekdays(Time)%in%c("Saturday","Sunday"),1,0)]
ds[(NumInstallation %in% c(2310320005,2341753003,2341753007,3324270001,3324270010))&
      (as.Date(Time)>ymd("2015-12-21")& as.Date(Time)<ymd("2016-01-04")),is_holiday:=1]
ds[(NumInstallation %in% c(2310320005,2341753003,2341753007,2341758001,2341757001))&
      (as.Date(Time)>ymd("2016-02-20")& as.Date(Time)<ymd("2016-03-07")),is_holiday:=1]
## the above 4 lines might need to be replaced by the 4 below (depend
## on the lubridate version)
## ds[(NumInstallation %in% c(2310320005,2341753003,2341753007,3324270001,3324270010))&
##       (Time>=ymd("2015-12-21")& Time<ymd("2016-01-04")),is_holiday:=1]
## ds[(NumInstallation %in% c(2310320005,2341753003,2341753007,2341758001,2341757001))&
##       (Time>=ymd("2016-02-20")& Time<ymd("2016-03-07")),is_holiday:=1]


## Number of hours of holiday inthe last XX hours before current point
CountHours <- function(var,n){    
    y <- rep(n,times=n-1)   
    #    for(i in n:length(var))y <- c(y,sum(var[(i-(n-1)):i])*var[i])
    for(i in n:length(var))y <- c(y,sum(var[(i-(n-1)):i]))
    y           
}
ds[order(Time,NumInstallation),Hol_6:=CountHours(is_holiday,6),by=NumInstallation]
ds[order(Time,NumInstallation),Hol_12:=CountHours(is_holiday,12),by=NumInstallation]
ds[order(Time,NumInstallation),Hol_24:=CountHours(is_holiday,24),by=NumInstallation]
ds[order(Time,NumInstallation),Hol_36:=CountHours(is_holiday,36),by=NumInstallation]
# here I multiply by is_holiday, otherwise, will be non zero too long
# after holiday ended.
ds[order(Time,NumInstallation),Hol_48:=CountHours(is_holiday,48)*is_holiday,by=NumInstallation]
ds[order(Time,NumInstallation),Hol_72:=CountHours(is_holiday,72)*is_holiday,by=NumInstallation]

## hour of the week
DayNum <- function(day){
  switch(day,
         Monday = 1,
         Tuesday = 2,
         Wednesday = 3,
         Thursday = 4,
         Friday = 5,
         Saturday = 6,
         Sunday = 7)
}
HourOfWeek <- function(t){
  dayid <- DayNum(weekdays(t))
  h <- (dayid-1)*24+hour(t)
}
ds[,':='(hweek=sapply(Time,function(x)HourOfWeek(x)))]

## Cleaning for 2341753007
# There is a huge constant plateau at the end of october. I will get
# rid of it to avoid pollution of the training set.
ds <- ds[!(NumInstallation==2341753007&
       Time>ymd_hms('2015-10-27 12:00:00')&
           Time<ymd_hms('2015-11-17 15:00:00'))]

## replace test set temperature by -99.
ds[Time>=testsetdates,Temperature:=-99]
## replace all missing values by -999 (some heat circuit measures are
## NA for some buildings.
ds[is.na(ds)] <- -999

# split in training and testing set.
inTest <- which(ds$Temperature==-99)
xtrain <- ds[-inTest]
xtest <- ds[inTest]


######################################################################
# train Model
######################################################################
# Xgboost parameters
param <- list(eval_metric="rmse",
              eta = .1,
              max_depth=16, 
              min_child_weight=1,
              subsample=1.0,
              colsample_bytree=1.0
              )

# convert training set to xgb format 
dtrain<-xgb.DMatrix(data=data.matrix(xtrain[,varnames, with=FALSE]),
                    label=data.matrix(xtrain[, Temperature]),
                    missing=-999)

# train model
xgb <- xgb.train(data = dtrain,
                 params = param,
                 nrounds = 50,
                 maximize=FALSE)

# convert test set to xgb format
dtest<-xgb.DMatrix(data=data.matrix(xtest[,varnames, with=FALSE]),
                   missing=-999)
# make prediction
p <- predict(xgb, dtest)
preds <- data.table(Time=xtest$Time,
                    NumInstallation=xtest$NumInstallation)
preds[,Temperature:=p]
preds <- preds[order(NumInstallation,Time)]

# I noticed that the temperature for build 9007420001 at the end of
# the training set was pretty constant. Seems they have set the target
# to 21C. I change the prediction for this building.
preds[NumInstallation==9007420001,Temperature:=21.0]
write.csv(preds, "submit25_2.csv", row.names=FALSE)


