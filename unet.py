
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( './image_0002.jpg' ),

    #block-001
    *to_Conv( name='b1i', s_filer="", n_filer=3, offset="(0,0,0)",width=2, height=40, depth=40, caption=" " ),
    *to_Conv( name='b1c', s_filer=256, n_filer=20, to="(b1i-east)", offset="(0,0,0)",width=4, height=40, depth=40, caption="Conv" ),
    *to_Pool(name='b1', offset="(0,0,0)", to="(b1c-east)", width=1, height=32, depth=32, opacity=0.5, caption=" "),

    
    *to_Conv( name='b2i', to="(b1-east)", s_filer="", n_filer=20, offset="(2,0,0)",width=4, height=32, depth=32, caption=" " ),
    *to_Conv( name='b2c', s_filer=128, n_filer=40, to="(b2i-east)", offset="(0,0,0)",width=6, height=32, depth=32, caption="Conv" ),
    *to_Pool(name='b2', offset="(0,0,0)", to="(b2c-east)", width=1, height=25, depth=25, opacity=0.5, caption=" "),

    *to_Conv( name='b3i', to="(b2-east)", s_filer="", n_filer=20, offset="(2,0,0)",width=6, height=25, depth=25, caption=" " ),
    *to_Conv( name='b3c', s_filer=64, n_filer=80, to="(b3i-east)", offset="(0,0,0)",width=8, height=25, depth=25, caption="Conv" ),
    *to_Pool(name='b3', offset="(0,0,0)", to="(b3c-east)", width=1, height=17, depth=17, opacity=0.5, caption=" "),

        *to_connection( 'b1','b2i' ),
      *to_connection( 'b2','b3i' ),

    #Decoder
    *to_Conv( name='b4', s_filer=32, n_filer=80, to="(b3-east)", offset="(2,0,0)",width=8, height=17, depth=17, caption=" " ),
    *to_Conv( name='b5', s_filer=64, n_filer=80, to="(b4-east)", offset="(0,0,0)",width=8, height=25, depth=25, caption="Conv" ),
    *to_Conv( name='b6', s_filer=64, n_filer=80, to="(b5-east)", offset="(2,0,0)",width=8, height=25, depth=25, caption=" " ),
    *to_Conv( name='b7', s_filer=128, n_filer=40, to="(b6-east)", offset="(0,0,0)",width=6, height=32, depth=32, caption="Conv" ),
   
    *to_Conv( name='b8', s_filer=128, n_filer=40, to="(b7-east)", offset="(2,0,0)",width=6, height=32, depth=32, caption="" ),
    *to_Conv( name='b9', s_filer=256, n_filer=20, to="(b8-east)", offset="(0,0,0)",width=4, height=40, depth=40, caption="Conv" ),


    *to_Conv( name='b10', s_filer="", n_filer=20, to="(b9-east)", offset="(2,0,0)",width=4, height=40, depth=40, caption="" ),
    *to_Conv( name='b11', s_filer=256, n_filer=2, to="(b10-east)", offset="(0,0,0)",width=2, height=40, depth=40, caption="Conv" ),
    *to_SoftMax( name='s1', s_filer=2, offset="(1,0,0)", to="(b11-east)", width=2, height=40, depth=40, opacity=0.8, caption="Softmax" ),

    *to_connection( 'b3','b4' ),
      *to_connection( 'b5','b6' ),
      *to_connection( 'b7','b8' ),
      *to_connection( 'b11','s1' ),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
