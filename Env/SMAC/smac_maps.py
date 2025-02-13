from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "25m": {
        "n_agents": 25,
        "n_enemies": 25,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m_vs_6m": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m_vs_11m": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "27m_vs_30m": {
        "n_agents": 27,
        "n_enemies": 30,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "MMM": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM2": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3z2s": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s2z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "7sz": {
        "n_agents": 14,
        "n_enemies": 14,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "5s10z": {
        "n_agents": 15,
        "n_enemies": 15,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s6z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s_vs_5z": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_3z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "1c3s5z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "1c3s5z_vs_1c3s6z": {
        "n_agents": 9,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "1c3s8z_vs_1c3s9z": {
        "n_agents": 12,
        "n_enemies": 13,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "corridor": {
        "n_agents": 6,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "6h_vs_8z": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
    },
    "2s_vs_1sc": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 300,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "so_many_baneling": {
        "n_agents": 7,
        "n_enemies": 32,
        "limit": 100,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 100,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "bane",
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
    "3m": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    
    # new maps
    "10m_vs_12m": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "15m_vs_17m": {
        "n_agents": 15,
        "n_enemies": 17,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "15m_vs_18m": {
        "n_agents": 15,
        "n_enemies": 18,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6s_vs_10z": {
        "n_agents": 6,
        "n_enemies": 10,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "6s_vs_11z": {
        "n_agents": 6,
        "n_enemies": 11,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "6s_vs_12z": {
        "n_agents": 6,
        "n_enemies": 12,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3c_vs_97zg": {
        "n_agents": 3,
        "n_enemies": 97,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
    "3c_vs_98zg": {
        "n_agents": 3,
        "n_enemies": 98,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
    "3c_vs_100zg": {
        "n_agents": 3,
        "n_enemies": 100,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
    "MMM30": {
        "n_agents": 10,
        "n_enemies": 13,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM31": {
        "n_agents": 10,
        "n_enemies": 13,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM32": {
        "n_agents": 10,
        "n_enemies": 14,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2z_vs_2z": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 80,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_zergling"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
    "1o_2r_vs_2r": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
    "5z_vs_1ul": {
        "n_agents": 5,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "zM_vs_z": {
        "n_agents": 4,
        "n_enemies": 3,
        "limit": 100,
        "a_race": "T",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "zMz"
    },
    "bZ_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
    "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
    "GMMM": {
        "n_agents": 13,
        "n_enemies": 13,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 4,
        "map_type": "GMMM",
    },
    "5z_vs_1ul": {
        "n_agents": 5,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },

    # configs for added marine battle maps
    "3m_vs_4m": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "3m_vs_5m": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4m": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4m_vs_5m": {
        "n_agents": 4,
        "n_enemies": 5,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4m_vs_6m": {
        "n_agents": 4,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m_vs_7m": {
        "n_agents": 6,
        "n_enemies": 7,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m_vs_8m": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m_vs_9m": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m_vs_10m": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m_vs_11m": {
        "n_agents": 9,
        "n_enemies": 11,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m_vs_8m": {
        "n_agents": 7,
        "n_enemies": 8,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m_vs_9m": {
        "n_agents": 7,
        "n_enemies": 9,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m": {
        "n_agents": 6, 
        "n_enemies": 6,
        "limit": 90,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 110,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m_vs_10m": {
        "n_agents": 9,
        "n_enemies": 10,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m_vs_7m": {
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m_vs_12m": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },

    # configs for added sz maps
    "1s1z": {
        "n_agents": 2,
        "n_enemies": 2,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s2z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1z2s": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2z1s": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s3z": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s2z": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_1s4z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s4z": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_3s2z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_3s3z": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_4s1z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_4s3z": {
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s5z": {
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s4z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_1s7z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_2s6z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s7z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s5z_vs_3s6z": {
        "n_agents": 7,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_4s5z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_5s3z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_5s5z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_6s2z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_7s1z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "4s7z": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "4s7z_vs_4s8z": {
        "n_agents": 11,
        "n_enemies": 12,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "4s7z_vs_4s9z": {
        "n_agents": 11,
        "n_enemies": 13,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "7s3z": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s9z": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s8z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "9s1z": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "8s2z": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s9z": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "8s3z": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s8z": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "8s1z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "9s2z": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s9z": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "5s2z_vs_5s3z": {
        "n_agents": 7,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "9s3z": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s8z": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "7s3z_vs_7s4z": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },

    # configs for added MMM Maps
    "MMM0": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM1": {
        "n_agents": 9,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM3": {
        "n_agents": 11,
        "n_enemies": 13,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM4": {
        "n_agents": 12,
        "n_enemies": 14,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM5": {
        "n_agents": 12,
        "n_enemies": 15,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM6": {
        "n_agents": 12,
        "n_enemies": 16,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },

    # 为丰富语义的MMM环境加难度
    "MMM_138vs159": {
        "n_agents": 12,
        "n_enemies": 15,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_138vs1410": {
        "n_agents": 12,
        "n_enemies": 15,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_238vs259": {
        "n_agents": 13,
        "n_enemies": 16,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_238vs2410": {
        "n_agents": 13,
        "n_enemies": 16,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs118": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs127": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs128": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs136": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs137": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs138": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs145": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs172": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs217": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs226": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs235": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM_127vs244": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "2s4z_vs_2s4z": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s4z_vs_4s2z": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s5z_vs_2s5z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s5z_vs_5s2z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s6z_vs_6s2z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s6z_vs_2s6z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s2z_vs_3s2z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s2z_vs_2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s3z_vs_3s3z": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s4z_vs_3s4z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s4z_vs_4s3z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
}



def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))
