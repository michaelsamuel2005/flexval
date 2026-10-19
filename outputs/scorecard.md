# flexval scorecard: 21 of 21 checks pass

| group | check | pass | detail |
|---|---|---|---|
| Options | Black-76 equals Monte Carlo for a European call (no jumps) | yes | Black-76 0.1883, MC 0.1875 +/- 0.0015 |
| Options | put-call parity | yes | C - P - df(F - K) = -2.13e-16 |
| Options | geometric control variate cuts the Asian standard error by half or more | yes | plain SE 0.00177, control-variate SE 0.00003 |
| Options | Asian call is worth less than the European call on the same forward and strike | yes | Asian 0.0441 < European 0.1883 |
| Storage | with zero volatility LP, LSMC and lattice agree to the dollar | yes | LP 33,795, LSMC 33,795, lattice 33,795 |
| Storage | extrinsic value is positive: LSMC above intrinsic | yes | LSMC 164,937 > intrinsic 34,667 |
| Storage | LSMC (a lower bound) does not exceed the lattice by more than 3 SE | yes | LSMC 164,937 +/- 2,968, lattice 169,508 |
| Storage | value rises with volatility | yes | sigma x1.5: 305,165 > 164,937 |
| Swing | with zero volatility LP and LSMC agree to the dollar | yes | LP -50,079, LSMC -50,079 |
| Swing | flexibility has value: LSMC above intrinsic | yes | 63,889 > -19,798 |
| Tolling | Kirk strip (with the OU terminal correlation) equals the Monte Carlo strip within 1% or 3 SE | yes | Kirk 51,301,665, MC 51,265,506 +/- 160,069 |
| Tolling | constraints cost value: constrained dispatch below the unconstrained strip | yes | constrained 53,235,269 <= strip 53,433,576 |
| Tolling | with no start cost and no minimum run the dispatch equals the strip exactly | yes | 53,433,576 vs 53,433,576 |
| PPA | cannibalisation lowers the capture rate and the value | yes | capture rate 0.880 with, 1.026 without (seasonal wind lifts it above 1) |
| PPA | the deal is worth zero at the breakeven strike | yes | value at breakeven -8,179 +/- 19,458 |
| Hedging | delta hedging cuts the P&L standard deviation | yes | unhedged sd 314,813, hedged daily 30,511 |
| Hedging | daily rebalancing beats monthly | yes | daily 30,511 < monthly 91,902 |
| Hedging | with jumps, the mean hedged loss equals the jump premium left out of the Black-76 price (within 3 SE) | yes | mean hedged P&L -311,181; jump premium 306,138 |
| Economics | the minimum fee makes the return on capital equal to the hurdle | yes | RoC 0.1200 at fee -81,113 |
| Desk book | every planted mispricing in the desk book is flagged | yes | 4 of 4 flagged |
| Desk book | no correctly booked deal is flagged | yes | false flags: [] |
