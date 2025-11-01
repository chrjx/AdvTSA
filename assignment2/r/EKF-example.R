##----------------------------------------------------------------
## EKF algorithm for use in Part 4 of computer exercise 2 in
## Advanced Time Series Analysis
##----------------------------------------------------------------

##----------------------------------------------------------------
## Do the simulation here and keep the values in y
##----------------------------------------------------------------


##----------------------------------------------------------------
## Estimation with the EKF
##----------------------------------------------------------------
## aInit : The starting guess of the AR coefficient estimate
aInit <- 0.5
## aVarInit : The initial variance for estimation of the AR coefficient
aVarInit <- 1
## sigma.v : Standard deviation of the system noise of x in the filter
sigma.v <- 1

## Initialize
## Init the state vector estimate
zt <- c(0,aInit)
## Init the variance matrices
Rv <- matrix(c(sigma.v^2,0,0,0), ncol=2)
## sigma.e : Standard deviation of the measurement noise in the filter
Re <- 1 

## Init the P matrix, that is the estimate of the state variance
Pt <- matrix(c(Re,0,0,aVarInit), nrow=2, ncol=2)
## The state is [X a] so the differentiated observation function is
Ht <- t(c(1,0))
## Init a vector for keeping the parameter a variance estimates
aVar <- rep(NA,length(y))
## and keeping the states
Z <- matrix(NA, nrow=length(y), ncol=2)
Z[,1] <- zt

## The Kalman filtering
for(t in 1:(length(y)-1))
  {
    ## The state transition function
    ft <- c(zt[2]*zt[1], zt[2])#ft <- c(Z[t,2]*Z[t,1], Z[t,2])
    ## The differentiated state transition function, Eq. (7.49)
    Ft <- matrix(c(zt[2],0,zt[1],1), ncol=2)
    ## Calculate the Kalman gain at time t, Eq. (7.46)
    Kt <- Ft %*% Pt %*% t(Ht) / c(Ht %*% Pt %*% t(Ht) + Re)
    ## Calculate the one-step prediction of the state, Eq. (7.45)
    zt = ft + Kt * (y[t] - zt[1])
    ## Calculate the prediction of the P matrix, Eq. (7.47) and (7.48)
    Pt <- Ft %*% Pt %*% t(Ft) + Rv - Kt %*% (Re + Ht %*% Pt %*% t(Ht)) %*% t(Kt)
    ## Keep the state estimate
    Z[t+1,] <- zt
    ## Keep the P[2,2], which is the variance of the estimate of a
    aVar[t+1] <- Pt[2,2]
  }
