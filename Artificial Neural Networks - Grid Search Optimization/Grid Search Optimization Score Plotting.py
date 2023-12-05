# Andrew S. Messecar, 2023

from pylab import plot , show , xlabel, ylabel, figure, xticks, title, legend
from numpy import arange

Number_Of_Layers = [ 1 , 2 , 3 ]
Layers = [ 1 , 2 ]

Plasma = [ 0.7484540507448546 , 0.789832468760481 , 0.8086116300042546]
MoS2 = [ 0.26328060208930565 , 0.3075262506830291 , 0.31824845937215]
GaN_Crystal = [ 0.2798734976396 , 0.419367764266158]
GaN_S2 = [ 0.14219884066305905 , 0.1954355805639633]
InN_Crystal = [ 0.2265330831632041 , 0.35836542671107846]

figure(dpi = 1200)
plot( Number_Of_Layers , Plasma , label = "Nitrogen Plasma")
plot( Number_Of_Layers , MoS2 , label = "MoS2")
plot( Layers , GaN_Crystal , label = "GaN Crystallinity")
plot( Layers , GaN_S2 , label = "GaN Lattice Ordering")
plot( Layers , InN_Crystal , label = "InN Crystallinity")
xlabel( "Number of Optimized Layers" )
ylabel( "Squared Error Validation Score" )
legend( loc = "center right")
title("Validation Score versus Optimized Layers")
xticks(arange(1.0, 3.1, step=1), ['1' , '2', '3'],rotation=0)
show()
