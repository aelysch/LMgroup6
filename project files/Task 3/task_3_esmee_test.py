import cv2
import numpy as np
import random
import time
from robobo_interface import SimulationRobobo

# Initialize parameters
alpha = 0.2  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration rate

# Final Q-table provided
Q = {
    ('none', 'none', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -31.423563326992856, 'left': -32.34926791475488, 'right': -31.239343019223472,
        'forward_left': -28.314732563066624, 'forward_right': -32.54223344097804, 'backward': -29.837182758743545,
        'sharp_left': -28.460017540850718, 'sharp_right': 13.236271599681933
    },
    ('none', 'none', 'none', 'none', 'high', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -2.378317187350443, 'right': 1.9044267194702131, 
        'forward_left': 0.16248524145304977, 'forward_right': 0, 'backward': -65.35642028014638, 
        'sharp_left': -10.81892014587375, 'sharp_right': -3.005796181652201
    },
    ('none', 'medium', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -10.595480395771727, 'left': 10.544269262263933, 'right': -11.15964899911744,
        'forward_left': -11.888581106206825, 'forward_right': -8.312503050325624, 'backward': -15.341854445910702,
        'sharp_left': -11.227134045073077, 'sharp_right': -13.08954972008516
    },
    ('none', 'none', 'none', 'none', 'none', 'low', 'no_collision'): {
        'forward': -12.69187084655728, 'left': -18.15776799021509, 'right': 23.287523062525977,
        'forward_left': -13.693714089846896, 'forward_right': -6.828781164771883, 'backward': -0.6402353562508059,
        'sharp_left': -14.183269470540772, 'sharp_right': -8.997506519848768
    },
    ('none', 'none', 'none', 'none', 'none', 'medium', 'no_collision'): {
        'forward': -1.0, 'left': -13.617990452371032, 'right': 26.88388406361899,
        'forward_left': -7.968123403894499, 'forward_right': 0, 'backward': 2.376,
        'sharp_left': -15.390398079171936, 'sharp_right': -10.884623858782934
    },
    ('none', 'none', 'none', 'low', 'none', 'none', 'no_collision'): {
        'forward': -8.828975470087638, 'left': -3.1818568712008446, 'right': -18.936165842770592,
        'forward_left': 3.0096698523386793, 'forward_right': -16.009195047357974, 'backward': -7.208685172250379,
        'sharp_left': -7.774046930655477, 'sharp_right': -17.04327932139349
    },
    ('none', 'medium', 'none', 'none', 'none', 'low', 'no_collision'): {
        'forward': -2.3960000000000004, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 2,
        'backward': 0, 'sharp_left': 0, 'sharp_right': -2.3364416
    },
    ('medium', 'none', 'none', 'none', 'low', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -3.462805693288269, 'right': -3.372922456095319,
        'forward_left': 0, 'forward_right': 0, 'backward': 0, 'sharp_left': 0, 'sharp_right': -4.041721995680485
    },
    ('medium', 'none', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -18.412506385941928, 'left': -9.536037573973864, 'right': -3.2793002543824,
        'forward_left': -8.73847779427783, 'forward_right': -16.587532848401054, 'backward': -10.599235401977003,
        'sharp_left': -2.50700594776586, 'sharp_right': -14.19610009531114
    },
    ('low', 'none', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -3.3228328585710063, 'left': -3.6, 'right': -5.013172513346764,
        'forward_left': -3.9191973412224295, 'forward_right': -8.682446664525063, 'backward': 0,
        'sharp_left': 2.5525004672000002, 'sharp_right': 0
    },
    ('high', 'high', 'none', 'none', 'low', 'none', 'no_collision'): {
        'forward': 2.0, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('high', 'high', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -4.121276697112705, 'left': -4.3755264, 'right': -6.5284408410561285,
        'forward_left': -3.6, 'forward_right': -0.5980000000000003, 'backward': 0,
        'sharp_left': -0.008060507896988867, 'sharp_right': 1
    },
    ('none', 'none', 'none', 'none', 'low', 'none', 'no_collision'): {
        'forward': 11.26324123922449, 'left': -21.018228528150285, 'right': -11.498959596613643,
        'forward_left': -11.824062991319831, 'forward_right': 6.868295008710682, 'backward': 9.967960447581728,
        'sharp_left': -15.296537144463606, 'sharp_right': -15.126499740332017
    },
    ('none', 'high', 'high', 'none', 'none', 'none', 'no_collision'): {
        'forward': -7.1021556148964065, 'left': -0.3368000000000002, 'right': -4.880000000000001,
        'forward_left': -8.317600689248524, 'forward_right': -2.0, 'backward': -2.5677717080555538,
        'sharp_left': -9.899122313524249, 'sharp_right': -0.25575029212124685
    },
    ('none', 'none', 'high', 'none', 'none', 'none', 'no_collision'): {
        'forward': -6.953805346811576, 'left': -11.464959860881077, 'right': -5.922019827200001,
        'forward_left': -7.609951321995073, 'forward_right': -11.597583716115444, 'backward': -7.961515683933284,
        'sharp_left': -2.6145551369159805, 'sharp_right': -12.53886072334418
    },
    ('none', 'none', 'none', 'low', 'low', 'none', 'no_collision'): {
        'forward': 1.3815812843967814, 'left': -0.030082050459982135, 'right': -9.04920026662598,
        'forward_left': 0, 'forward_right': 0.6268726510381333, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -6.0897108852882464
    },
    ('none', 'none', 'none', 'medium', 'none', 'none', 'no_collision'): {
        'forward': -7.002030873556777, 'left': 12.470303911679682, 'right': -7.4955600652526515,
        'forward_left': 5.502512085490511, 'forward_right': -5.6227556193398005, 'backward': -1.7820301364892517,
        'sharp_left': 0, 'sharp_right': -13.773041246066077
    },
    ('none', 'none', 'none', 'high', 'low', 'none', 'no_collision'): {
        'forward': -6.19004551004126, 'left': 7.0, 'right': 0, 'forward_left': 0,
        'forward_right': 0, 'backward': 0, 'sharp_left': 5.575710194881766, 'sharp_right': 0
    },
    ('none', 'low', 'none', 'none', 'high', 'none', 'no_collision'): {
        'forward': 4.66, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 2.355849535646778
    },
    ('none', 'none', 'medium', 'none', 'none', 'none', 'no_collision'): {
        'forward': -10.747483614048083, 'left': -15.739509076456683, 'right': -9.56733561211912,
        'forward_left': -13.778668215564677, 'forward_right': -12.129988271240194, 'backward': -9.492637726766269,
        'sharp_left': -6.904906926697439, 'sharp_right': -15.545718206290498
    },
    ('none', 'none', 'low', 'none', 'none', 'none', 'no_collision'): {
        'forward': -7.9459527102549075, 'left': -6.193459572816081, 'right': 0,
        'forward_left': -2.5727779797484622, 'forward_right': 0, 'backward': -7.90944010000591,
        'sharp_left': 0, 'sharp_right': 0.5824610114785193
    },
    ('none', 'low', 'medium', 'none', 'none', 'none', 'no_collision'): {
        'forward': -3.6, 'left': -6.687561988072796, 'right': 0, 'forward_left': -2.0,
        'forward_right': 0, 'backward': -2.9409168598127238, 'sharp_left': 1, 'sharp_right': 0
    },
    ('high', 'none', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -7.959097992900551, 'left': -7.219847751812495, 'right': -9.218395451271473,
        'forward_left': -10.18847038797846, 'forward_right': -7.595810139784129, 'backward': -8.801693390914116,
        'sharp_left': -8.313436034282038, 'sharp_right': -9.72449205404508
    },
    ('none', 'low', 'none', 'low', 'low', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0, 'forward_left': 2.0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'low', 'none', 'high', 'none', 'no_collision'): {
        'forward': -2.0, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 0.33,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'low', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -3.8188340243732384, 'left': 27.598811777218305, 'right': 0,
        'forward_left': -4.073633705878424, 'forward_right': -2.8179043955897423, 'backward': 0,
        'sharp_left': -7.332838350668142, 'sharp_right': -8.458336998583938
    },
    ('none', 'none', 'none', 'none', 'medium', 'none', 'no_collision'): {
        'forward': 41.82355782442466, 'left': -2.5647523869227475, 'right': 1.3813845697649758,
        'forward_left': 10.525226602299567, 'forward_right': 7.259700131840001, 'backward': 13.234808342498226,
        'sharp_left': -5.459685323057798, 'sharp_right': -3.4704101035163126
    },
    ('none', 'none', 'none', 'none', 'none', 'high', 'no_collision'): {
        'forward': -8.01145094421614, 'left': -11.670510719977447, 'right': 40.68655408594542,
        'forward_left': -4.526741103750945, 'forward_right': 0, 'backward': 5.695342163046401,
        'sharp_left': 0, 'sharp_right': -4.279455805901667
    },
    ('none', 'none', 'none', 'none', 'high', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 65.94150015587873, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': -1.62390991165155
    },
    ('none', 'none', 'none', 'low', 'medium', 'none', 'no_collision'): {
        'forward': 6.7047473766186805, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': -5.191572483163757, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'high', 'medium', 'none', 'no_collision'): {
        'forward': -7.523440031351908, 'left': 8.654061184049418, 'right': 0,
        'forward_left': -1.1552921176553155, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -2.909990813015421
    },
    ('none', 'none', 'none', 'medium', 'high', 'none', 'no_collision'): {
        'forward': -4.235287293852818, 'left': 0, 'right': 0,
        'forward_left': 5.808400461207926, 'forward_right': -4.6702666728453766, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'medium', 'medium', 'none', 'no_collision'): {
        'forward': 7.290315039987812, 'left': 2.016264865193984, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -7.301740527912873
    },
    ('none', 'none', 'none', 'high', 'none', 'none', 'no_collision'): {
        'forward': -15.237560026426788, 'left': 37.04402082691901, 'right': -7.009915218295484,
        'forward_left': 0, 'forward_right': -7.049622236781454, 'backward': 6.035807167917358,
        'sharp_left': 0, 'sharp_right': -4.104714614663711
    },
    ('low', 'none', 'none', 'none', 'medium', 'medium', 'no_collision'): {
        'forward': 6.0, 'left': 0, 'right': 0, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'none', 'low', 'high', 'no_collision'): {
        'forward': -3.468653070119148, 'left': -6.169956625966399, 'right': 2.0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'medium', 'none', 'low', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': 0.33333333, 'right': 0, 'forward_left': 0, 'forward_right': 0,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'medium', 'medium', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -7.404084954109172, 'right': 3.565903380961877,
        'forward_left': -2.708075276923539, 'forward_right': -3.156595871390881,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'high', 'medium', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -13.21053722982385, 'right': -2.7061370227999344,
        'forward_left': -2.0, 'forward_right': -2.0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 6
    },
    ('none', 'none', 'none', 'none', 'medium', 'high', 'no_collision'): {
        'forward': 0, 'left': -8.15412423569447, 'right': 2,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': -6.075058642048106, 'sharp_right': 0
    },
    ('none', 'medium', 'none', 'high', 'low', 'none', 'no_collision'): {
        'forward': -0.8120000000000002, 'left': 6.34522333, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'low', 'none', 'high', 'low', 'low', 'no_collision'): {
        'forward': -2.0, 'left': 2, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'low', 'high', 'none', 'no_collision'): {
        'forward': 2.384694850659157, 'left': 2.240312915531106, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'none', 'low', 'medium', 'no_collision'): {
        'forward': 8.5843456, 'left': 0, 'right': 0,
        'forward_left': -0.42763693518235135, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'high', 'high', 'none', 'no_collision'): {
        'forward': -7.572161867622518, 'left': 4.8, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'medium', 'low', 'none', 'no_collision'): {
        'forward': -3.703424711455016, 'left': 8.115248947239534, 'right': -7.073000344553162,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('medium', 'medium', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -4.456614445198343, 'right': -1.2121838473230855,
        'forward_left': -2.3960000000000004, 'forward_right': -5.387972276214828,
        'backward': 0, 'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'high', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -3.6, 'left': -10.838220269341354, 'right': -6.313146073013913,
        'forward_left': -2.0, 'forward_right': -3.6, 'backward': -5.727316377503097,
        'sharp_left': -6.646202937904907, 'sharp_right': -4.594555513828964
    },
    ('none', 'medium', 'high', 'none', 'none', 'none', 'no_collision'): {
        'forward': -6.15744, 'left': -6.825427230559183, 'right': -2.0,
        'forward_left': -1.9390775589447267, 'forward_right': -2.0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -5.474302905507032
    },
    ('high', 'none', 'none', 'none', 'none', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': .5678982835845447,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -0.5678982835845447
    },
    ('none', 'medium', 'low', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -5.811673859339955, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 1
    },
    ('medium', 'low', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.2662803916030576, 'left': 0, 'right': 0,
        'forward_left': -2.3914610543344472, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 1
    },
    ('none', 'none', 'none', 'none', 'medium', 'medium', 'no_collision'): {
        'forward': -1.6879253586895862, 'left': 0, 'right': 4.083109938591587,
        'forward_left': -1.4657075786307185, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'low', 'low', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.4522925882787843, 'left': -5.123980938434021, 'right': 2.7682400000000005,
        'forward_left': -3.538611137270294, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('medium', 'none', 'none', 'none', 'none', 'low', 'no_collision'): {
        'forward': -0.9109582506720746, 'left': 0, 'right': 0,
        'forward_left': -2.7861997670381675, 'forward_right': 0, 'backward': 0,
        'sharp_left': -3.7901064375863083, 'sharp_right': 7.889
    },
    ('high', 'medium', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -1.604, 'left': -4.622901969738573, 'right': 0,
        'forward_left': -2.3960000000000004, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('low', 'high', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 2.0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('high', 'low', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': -2.0, 'right': 5.0,
        'forward_left': 0, 'forward_right': -2.0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('high', 'high', 'high', 'none', 'none', 'none', 'no_collision'): {
        'forward': -10.878838483943362, 'left': 4.077376200285674, 'right': -2.8314722662400005,
        'forward_left': -5.836696881165313, 'forward_right': 0, 'backward': -2.0,
        'sharp_left': -5.465357504932251, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'none', 'low', 'low', 'no_collision'): {
        'forward': -3.495101173863954, 'left': -6.477158156371326, 'right': -0.44617327201464313,
        'forward_left': -0.5711008547174999, 'forward_right': 15.33941207175096, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 3.583521768811138
    },
    ('none', 'none', 'none', 'medium', 'none', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 6.134531456, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'none', 'medium', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 24.107775126816357, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -7.2046078191755845
    },
    ('medium', 'none', 'none', 'none', 'none', 'medium', 'no_collision'): {
        'forward': -3.440119613177435, 'left': 0, 'right': 4,
        'forward_left': 0, 'forward_right': 0, 'backward': 0.0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'high', 'none', 'none', 'low', 'none', 'no_collision'): {
        'forward': 2.0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'medium', 'none', 'none', 'low', 'no_collision'): {
        'forward': 0, 'left': -4.682683870787004, 'right': 1,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('medium', 'none', 'none', 'none', 'medium', 'none', 'no_collision'): {
        'forward': .066642291, 'left': 0, 'right': -1.377341260013371,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('low', 'none', 'none', 'none', 'high', 'none', 'no_collision'): {
        'forward': 1, 'left': 0, 'right': -6.25681102116671,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'low', 'none', 'none', 'medium', 'low', 'no_collision'): {
        'forward': -2.8388989721922577, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 2.9, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'medium', 'none', 'medium', 'medium', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 2.0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('low', 'medium', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': 0, 'right': 6.134531456,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'low', 'none', 'high', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': -6.139584778624393, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 6.134531456
    },
    ('high', 'high', 'none', 'none', 'none', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 2.0, 'backward': -2.0,
        'sharp_left': 0, 'sharp_right': -5.8689921370362335
    },
    ('medium', 'high', 'none', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': -2.7668372816718976, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -4.996297034901847
    },
    ('none', 'low', 'high', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.586567123864689, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 6.134531456
    },
    ('high', 'high', 'medium', 'none', 'none', 'none', 'no_collision'): {
        'forward': -3.4550377829521537, 'left': -2.4510578000280643, 'right': 0.2019627345056315,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'none', 'none', 'high', 'medium', 'no_collision'): {
        'forward': 1, 'left': -3.4252201663128967, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('high', 'none', 'none', 'none', 'low', 'none', 'no_collision'): {
        'forward': 2.0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('high', 'none', 'none', 'low', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 5.927, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -5.935371402577113
    },
    ('none', 'high', 'high', 'none', 'none', 'low', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': -2.9662400000000004,
        'forward_left': 0, 'forward_right': 2.9372, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'medium', 'low', 'none', 'none', 'no_collision'): {
        'forward': -3.5075165607232925, 'left': 0, 'right': 0,
        'forward_left': 6.134531456, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'medium', 'none', 'high', 'none', 'no_collision'): {
        'forward': -3.6489956437044917, 'left': 6.134531456, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('medium', 'high', 'low', 'none', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 2.0
    },
    ('high', 'high', 'high', 'none', 'low', 'none', 'no_collision'): {
        'forward': 6.227798560552198, 'left': 0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'medium', 'high', 'low', 'none', 'none', 'no_collision'): {
        'forward': -2.0, 'left': 0.62, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'low', 'medium', 'medium', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 3.6,
        'forward_left': 6.134531456, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'low', 'none', 'high', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 2.0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 2.1, 'sharp_right': 0
    },
    ('none', 'none', 'medium', 'high', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': 3.0, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 4.0,
        'sharp_left': 0, 'sharp_right': -2.415903284821094
    },
    ('none', 'none', 'low', 'medium', 'low', 'none', 'no_collision'): {
        'forward': 0, 'left': 0, 'right': 0,
        'forward_left': 2.0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'low', 'high', 'none', 'none', 'no_collision'): {
        'forward': 0, 'left': 5.0, 'right': -2.0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': -2.0
    },
    ('none', 'none', 'low', 'medium', 'high', 'low', 'no_collision'): {
        'forward': -3.8249425717826564, 'left': 0, 'right': 0,
        'forward_left': 3.99972920, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    },
    ('none', 'none', 'low', 'high', 'medium', 'none', 'no_collision'): {
        'forward': 0, 'left': 10.396, 'right': 0,
        'forward_left': 0, 'forward_right': 0, 'backward': 0,
        'sharp_left': 0, 'sharp_right': 0
    }
}




# Action space
actions = ["forward", "left", "right", "forward_left", "forward_right", "sharp_left", "sharp_right"]


def preprocess_camera_data(rob):
    """
    Detect green and red areas in the camera feed and divide the screen into left, center, and right sections.
    Returns the percentage of green and red in each section.
    """
    # Read camera image
    image = rob.read_image_front()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for green and red
    lower_green = np.array([40, 50, 50], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for green and red
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Divide the screen into three sections: left, center, right
    height, width = green_mask.shape
    sections = {
        "green": {
            "left": green_mask[:, :width // 3].sum() // 255,
            "center": green_mask[:, width // 3:2 * width // 3].sum() // 255,
            "right": green_mask[:, 2 * width // 3:].sum() // 255,
        },
        "red": {
            "left": red_mask[:, :width // 3].sum() // 255,
            "center": red_mask[:, width // 3:2 * width // 3].sum() // 255,
            "right": red_mask[:, 2 * width // 3:].sum() // 255,
        }
    }

    total_pixels = green_mask.size

    # Convert to percentages
    for color in sections:
        for section in sections[color]:
            sections[color][section] = (sections[color][section] / total_pixels) * 100

    return sections["green"], sections["red"]

def get_discrete_state(rob):
    """
    Get the discrete state for green and red areas in left, center, and right sections.
    """
    green_percentages, red_percentages = preprocess_camera_data(rob)

    def classify_green(percentage):
        """Classifies green intensity based on default thresholds."""
        if percentage > 0.5:
            return "high"
        elif percentage > 0.1:
            return "medium"
        elif percentage > 0.05:
            return "low"
        else:
            return "none"

    def classify_red(percentage):
        """Classifies red intensity based on new thresholds (high >4, medium >2, low >1, else none)."""
        if percentage > 5:
            return "high"
        elif percentage > 2:
            return "medium"
        elif percentage > 0.0005: 
            return "low"
        else:
            return "none"

    green_state = {section: classify_green(green_percentages[section]) for section in green_percentages}
    red_state = {section: classify_red(red_percentages[section]) for section in red_percentages}

    collision_detected = rob.read_irs()[4] > 10000
    collision_state = "collision" if collision_detected else "no_collision"
    print(green_state["left"], green_state["center"], green_state["right"],
        red_state["left"], red_state["center"], red_state["right"],
        collision_state)

    return (
        green_state["left"], green_state["center"], green_state["right"],
        red_state["left"], red_state["center"], red_state["right"],
        collision_state
    )

def execute_action(rob, action):
    """
    Execute the given action on the robot.
    """
    if action == "forward":
        rob.move_blocking(30, 30, 500)
    elif action == "left":
        #rob.move_blocking(-50, -50, 100)
        rob.move_blocking(-5, 5, 200)
    elif action == "right":
        #rob.move_blocking(-50, -50, 100)
        rob.move_blocking(5, -5, 200)
    elif action == "forward_left":
        rob.move_blocking(30, 50, 500)
    elif action == "forward_right":
        rob.move_blocking(50, 30, 500)
    # elif action == "backward":
    #     rob.move_blocking(-100, -100, 500)
    elif action == "sharp_left":
        rob.move_blocking(-10, 10, 1000)
    elif action == "sharp_right":
        rob.move_blocking(10, -10, 1000)
    else:
        print(f"Unknown action: {action}")

def run_sequence(rob, max_duration=120):
    """Run a demonstration using the final Q-table to show off the learned model."""
    rob.set_phone_tilt(103,50)
    state = get_discrete_state(rob)
    total_reward = 0
    collected_blocks = 0
    start_time = time.time()


    print("\n[Demo] Executing final policy using the learned Q-table.")
    while time.time() - start_time < max_duration:
        # Choose the best action for the current state
        if state in Q:
            action = max(Q[state], key=Q[state].get)  # Greedy action
        else:
            action = random.choice(["left", "right"])  # Randomly choose left or right  # Default action if the state is unknown

        # else:
        #     action = "forward"  # Default action if the state is unknown

        print(f"[Action] Executing action: {action}")
        
        # Execute the action
        execute_action(rob, action)  

        # Collision detection
        collision_detected = rob.read_irs()[4] > 10000

        # Get the next state
        next_state = get_discrete_state(rob)
        # collected = collision_detected and next_state[3] == "high"

        # Track collected blocks
        # if collected:
        #     collected_blocks += 1
        #     print(f"[Demo] Collected green box! Total collected: {collected_blocks}")

        state = next_state

    print(f"[Demo] Total Reward: {total_reward}, Blocks Collected: {collected_blocks}")
    return total_reward, collected_blocks

def set_tilt(rob):
    rob.set_phone_tilt(97,50)


def run_demo(rob):
    """Set up the robot and run the demonstration."""
    rob = SimulationRobobo()
    rob.play_simulation()
    
    try:
        total_reward, collected_blocks = run_sequence(rob, max_duration=180)
        print(f"Demo Finished: Total Reward = {total_reward}, Blocks Collected = {collected_blocks}")
    finally:
        rob.stop_simulation()
