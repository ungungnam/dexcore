/* Chart data of the page. Generated file: do not edit by hand. */
window.DEXCORE_FIGURES = {
 "figures": {
  "pub_obs1_variation": {
   "type": "bar",
   "title": "Contact variation across time and across object or subject identity, relative to two takes",
   "panels": [
    {
     "title": "TACO",
     "x": {
      "label": "what is compared",
      "categories": [
       "amount over time",
       "pattern over time",
       "pattern across object meshes"
      ]
     },
     "y": {
      "label": "contact variation, relative to two takes of the same object and action",
      "direction": "none"
     },
     "series": [
      {
       "name": "ratio to the distance between two takes",
       "values": [
        2.1235419511795044,
        1.035969227552414,
        1.4829091131687164
       ]
      }
     ],
     "refs": [
      {
       "axis": "y",
       "value": 1.0,
       "label": "two takes, same object and action"
      }
     ]
    },
    {
     "title": "ARCTIC",
     "x": {
      "label": "what is compared",
      "categories": [
       "amount over time",
       "pattern over time",
       "pattern across subjects (vs two takes of one subject)"
      ]
     },
     "y": {
      "label": "contact variation, relative to two takes of the same object and action",
      "direction": "none"
     },
     "series": [
      {
       "name": "ratio to the distance between two takes",
       "values": [
        1.0987825258211656,
        0.9405823864720084,
        1.155891857363961
       ]
      }
     ],
     "refs": [
      {
       "axis": "y",
       "value": 1.0,
       "label": "two takes, same object and action"
      }
     ]
    },
    {
     "title": "OakInk2",
     "x": {
      "label": "what is compared",
      "categories": [
       "amount over time",
       "pattern over time",
       "pattern across object instances"
      ]
     },
     "y": {
      "label": "contact variation, relative to two takes of the same object and action",
      "direction": "none"
     },
     "series": [
      {
       "name": "ratio to the distance between two takes",
       "values": [
        0.6013318006044779,
        0.4521745386031958,
        1.6860250900773441
       ]
      }
     ],
     "refs": [
      {
       "axis": "y",
       "value": 1.0,
       "label": "two takes, same object and action"
      }
     ]
    }
   ],
   "note": "Each bar is a ratio of two mean distances between pairs of contact maps. Amount: the absolute difference of the summed contact values. Pattern: the L2 distance between maps normalised to sum 1. The numerator compares early and late frames of one take (across time), or frames of another object mesh (TACO), subject (ARCTIC) or object instance (OakInk2) at the same phase of the motion. The denominator compares two takes of the same object and action at the same phase, so 1 means as different as two takes. No direction is better. Bars are means of per-group ratios; groups per bar: TACO 8, 8, 8; ARCTIC 22, 22, 22; OakInk2 39, 39, 17. There are no error bars: intervals exist per group only. Amount and pattern are different measures, each divided by its own take-to-take level. On ARCTIC the third bar is relative to two takes of the same subject, which is not the denominator of its first two bars; against that denominator it is 1.011. On TACO an object mesh is confounded with the people who used it. What a take contains differs: TACO takes include approach and release, ARCTIC takes are long and hold several episodes, OakInk2 segments are interaction only."
  },
  "pub_obs2_capability_by_class": {
   "type": "bar",
   "title": "How often the modelled wrench capability rises by more than 25%, by kind of contact event",
   "panels": [
    {
     "title": "TACO",
     "x": {
      "label": "kind of contact event (number of test events)",
      "categories": [
       "contact moves (119)",
       "moves, amount changes (435)",
       "mainly amount changes (72)"
      ]
     },
     "y": {
      "label": "share of events that gain more than 25% modelled capability",
      "direction": "none"
     },
     "series": [
      {
       "name": "events that gain more than 25% capability",
       "values": [
        0.20168067226890757,
        0.4574712643678161,
        0.6111111111111112
       ]
      }
     ],
     "highlight": {
      "category": "contact moves (119)",
      "label": "reconfiguration"
     }
    },
    {
     "title": "ARCTIC",
     "x": {
      "label": "kind of contact event (number of test events)",
      "categories": [
       "contact moves (26)",
       "moves, amount changes (118)",
       "mainly amount changes (40)"
      ]
     },
     "y": {
      "label": "share of events that gain more than 25% modelled capability",
      "direction": "none"
     },
     "series": [
      {
       "name": "events that gain more than 25% capability",
       "values": [
        0.23076923076923078,
        0.4576271186440678,
        0.475
       ]
      }
     ],
     "highlight": {
      "category": "contact moves (26)",
      "label": "reconfiguration"
     }
    }
   ],
   "note": "Share of test events whose modelled wrench capability (the mean capacity of a grasp model over 76 force and torque directions) gains more than 25%, for three kinds of contact event. Contact moves: a persistent re-arrangement of where the hand touches at a roughly constant contact amount, inside a maintained grasp (the reconfigurations of this observation). Moves, amount changes: persistent events that also change the contact amount. Mainly amount changes: events that are mostly a change of the contact amount. Onsets, releases and short-lived changes are not shown. Events per kind, in axis order: TACO 119, 435, 72; ARCTIC 26, 118, 40. There are no error bars: no interval was computed for these shares, so this is a descriptive comparison."
  },
  "pub_obs3_future_ladder": {
   "type": "line",
   "title": "Predicting the future structured contact from each contact description",
   "panels": [
    {
     "title": "TACO",
     "x": {
      "label": "contact description of a frame, ordered by size",
      "categories": [
       "parts",
       "+ amount",
       "+ position & direction",
       "+ spread & patch count",
       "compact + coarse hand",
       "compact + both",
       "dense map",
       "dense map + hand surface points"
      ]
     },
     "y": {
      "label": "future-structure error relative to the full dense state, the last level (ratio − 1)",
      "direction": "lower_better"
     },
     "series": [
      {
       "name": "error of predicting the future structured contact",
       "values": [
        0.1534972360460254,
        -0.010674228742894698,
        -0.182430350124546,
        -0.18711284650407622,
        -0.20983083341989295,
        -0.21371236823060313,
        0.13300508239042874,
        0.0
       ],
       "lo": [
        0.13469098613202812,
        -0.025060408019462087,
        -0.19375948404682627,
        -0.1981871182739633,
        -0.2219882171614788,
        -0.22560570784934014,
        0.11637222331341876,
        0.0
       ],
       "hi": [
        0.17223557680781007,
        0.005015048185461345,
        -0.17052227432157435,
        -0.17482206806564962,
        -0.1987488109692727,
        -0.20290190308747097,
        0.15076021875655002,
        0.0
       ]
      }
     ],
     "highlight": {
      "category": "+ position & direction",
      "label": "compact state: 48 numbers per frame"
     }
    },
    {
     "title": "ARCTIC",
     "x": {
      "label": "contact description of a frame, ordered by size",
      "categories": [
       "parts",
       "+ amount",
       "+ position & direction",
       "+ spread & patch count",
       "compact + coarse hand",
       "compact + both",
       "dense map",
       "dense map + hand surface points"
      ]
     },
     "y": {
      "label": "future-structure error relative to the full dense state, the last level (ratio − 1)",
      "direction": "lower_better"
     },
     "series": [
      {
       "name": "error of predicting the future structured contact",
       "values": [
        0.22843789939265835,
        0.015498806256172948,
        -0.3196839301489157,
        -0.31865724865789036,
        -0.3248173304254077,
        -0.3313186493804553,
        0.08088240427944604,
        0.0
       ],
       "lo": [
        0.17999170800765252,
        -0.02682301819950259,
        -0.35270693042652385,
        -0.35079237117578155,
        -0.35221734374780994,
        -0.3579039622098921,
        0.06322874686479028,
        0.0
       ],
       "hi": [
        0.29073538615446715,
        0.06345537544912311,
        -0.2867932462125389,
        -0.28578871343250223,
        -0.29925919398391043,
        -0.3052108051909519,
        0.09889964421113166,
        0.0
       ]
      }
     ],
     "highlight": {
      "category": "+ position & direction",
      "label": "compact state: 48 numbers per frame"
     }
    }
   ],
   "note": "Error of predicting the structured contact state 1, 4 and 8 frames ahead from the last 8 frames of each description, relative to the same prediction from the full dense state (the last level: the dense contact map plus hand surface points, 0 by definition). Plotted is the mean error ratio over the predicted quantities (which parts touch, contact amount, position and direction, wrench profile) and over the three horizons, minus 1: negative values are lower errors, and lower is better. Levels, in axis order: which hand parts touch; plus how much contact each part makes; plus where on the object each part touches and which way that surface faces (the compact state, highlighted); the compact state plus the spread and the number of contact patches of each part; the compact state plus a coarse hand pose; the compact state plus both; the dense contact map; the dense contact map plus hand surface points. The levels are ordered by size, not nested, and the two dense levels describe the same frame differently: they do not add to the compact ones. Means of 3 fits; test takes: TACO 186, ARCTIC 44; error bars are 95% paired bootstrap intervals over the test takes. All levels go through one fixed small prediction model, so the dense inputs doing worse is a statement about that model, not about the information in the dense state. Every model also receives an object descriptor and the local object trajectory, including the future states of the object. A comparison between inputs, not an absolute forecasting result."
  }
 }
};
