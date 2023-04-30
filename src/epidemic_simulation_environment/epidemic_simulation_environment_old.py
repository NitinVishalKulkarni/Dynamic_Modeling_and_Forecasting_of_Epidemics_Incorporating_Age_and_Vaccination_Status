# Imports
from typing import Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


# Defining the Epidemic Simulation Environment.
# noinspection DuplicatedCode
class EpidemicSimulation(gym.Env):
    """This class implements the Disease Mitigation environment."""

    def __init__(self, env_config):

        """This method initializes the environment.

        :param env_config: Dictionary containing the configuration for environment initialization."""

        self.data_path = env_config['data_path']
        self.state_name = env_config['state_name']

        self.covid_data = pd.read_csv(f'{self.data_path}/{self.state_name}.csv')
        self.covid_data['date'] = pd.to_datetime(self.covid_data['date'])
        self.population = env_config['state_population']
        self.start_date = env_config['start_date']
        # self.observation_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=11, shape=(4,))
        self.action_space = spaces.Discrete(12)

        # Population Dynamics by Epidemiological Compartments:
        self.number_of_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible'].iloc[0]
        self.number_of_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed'].iloc[0]
        self.number_of_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected'].iloc[0]
        self.number_of_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized'].iloc[0]
        self.number_of_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered'].iloc[0]
        self.number_of_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased'].iloc[0]

        # Population Dynamics by Vaccination Status:
        self.number_of_unvaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'unvaccinated_individuals'].iloc[0]
        self.number_of_fully_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'fully_vaccinated_individuals'].iloc[0]
        self.number_of_booster_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'boosted_individuals'].iloc[0]

        # Susceptible Compartment by Vaccination Status:
        self.number_of_unvaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_UV'].iloc[0]
        self.number_of_fully_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_FV'].iloc[0]
        self.number_of_booster_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_BV'].iloc[0]

        # Exposed Compartment by Vaccination Status:
        self.number_of_unvaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_UV'].iloc[0]
        self.number_of_fully_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_FV'].iloc[0]
        self.number_of_booster_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_BV'].iloc[0]

        # Infected Compartment by Vaccination Status:
        self.number_of_unvaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_UV'].iloc[0]
        self.number_of_fully_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_FV'].iloc[0]
        self.number_of_booster_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_BV'].iloc[0]

        # Hospitalized Compartment by Vaccination Status:
        self.number_of_unvaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_UV'].iloc[0]
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_FV'].iloc[0]
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_BV'].iloc[0]

        # Recovered Compartment by Vaccination Status:
        self.number_of_unvaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_UV'].iloc[0]
        self.number_of_fully_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_FV'].iloc[0]
        self.number_of_booster_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_BV'].iloc[0]

        # Deceased Compartment by Vaccination Status:
        self.number_of_unvaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_UV'].iloc[0]
        self.number_of_fully_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_FV'].iloc[0]
        self.number_of_booster_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_BV'].iloc[0]

        # LISTS FOR PLOTTING:
        # Lists by Epidemiological Compartments:
        self.number_of_susceptible_individuals_list = [self.number_of_susceptible_individuals]
        self.number_of_exposed_individuals_list = [self.number_of_exposed_individuals]
        self.number_of_infected_individuals_list = [self.number_of_infected_individuals]
        self.number_of_hospitalized_individuals_list = [self.number_of_hospitalized_individuals]
        self.number_of_recovered_individuals_list = [self.number_of_recovered_individuals]
        self.number_of_deceased_individuals_list = [self.number_of_deceased_individuals]

        # Lists by Vaccination Status:
        self.number_of_unvaccinated_individuals_list = [self.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = [self.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = [self.number_of_booster_vaccinated_individuals]

        # Lists for Epidemiological Compartments by Vaccination Status:
        # Susceptible Compartment
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.number_of_booster_vaccinated_susceptible_individuals]

        # Exposed Compartment
        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.number_of_booster_vaccinated_exposed_individuals]

        # Infected Compartment
        self.number_of_unvaccinated_infected_individuals_list = \
            [self.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.number_of_booster_vaccinated_infected_individuals]

        # Hospitalized Compartment
        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.number_of_booster_vaccinated_hospitalized_individuals]

        # Recovered Compartment
        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.number_of_booster_vaccinated_recovered_individuals]

        # Deceased Compartment
        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.number_of_booster_vaccinated_deceased_individuals]

        self.economic_and_public_perception_rate = 100.0
        self.economic_and_public_perception_rate_list = [self.economic_and_public_perception_rate]

        # Values of the epidemiological model parameters:
        self.alpha = [0.9921061737800656, 0.999999999999382, 0.9997006558362771, 0.9968623277813831,
                      0.9999999976899998, 0.9999999999934777, 0.9999832708276221, 0.9999977204877676,
                      0.999990484796472, 0.9999998603343343, 0.9999994613134795, 0.8667971932071317,
                      0.9987829564173087, 0.9242816529512278, 0.8913601714594757]

        # Exposure Rates
        self.beta_original = [4.979203079794559, 4.97458653699875, 4.998978463651875, 4.150487813293975,
                              4.999575013079198, 4.99997112741788, 4.878199450296301, 4.902405795261032,
                              4.999999999999789, 3.3489137022430944, 4.999972601990617, 3.6779583437607575,
                              1.6469992682905124, 2.129861260361653, 4.204429178608633]
        self.beta = None

        # Hospitalization Rates:
        self.delta_uv = [0.0025502796914869566, 1.1927977532355527e-14, 0.00033590586973677593, 0.004439027683949327,
                         0.004444439879977823, 0.004438688228942953, 0.003410784678446129, 0.0019063365305083818,
                         0.002738613153107736, 0.0031973457209689992, 0.0012563474592947861, 0.0015391354807226135,
                         0.0003191149877727051, 0.0028976182507645257, 0.0009689996872575844]

        self.delta_fv = [0.0013644550830824043, 7.998410292233626e-10, 0.004441947264565481, 0.00310033406963474,
                         0.003111528871142363, 0.0021878944815369207, 0.003842754738498646, 3.0471233519136476e-05,
                         0.0008992145473468121, 0.000546659225923523, 0.0004165955972560249, 0.00040581377314839235,
                         0.0012433423734815078, 0.0029025653261004203, 0.0013589929204257342]

        self.delta_bv = [0.000516666, 0.000516666, 0.000516666, 0.000516666, 0.000516666, 0.000516666,
                         0.004435513215501976, 0.0008285269025012719, 0.0010351714630686705, 0.00035536520680206913,
                         0.000436249724771229, 0.0003222169211899946, 0.00028863269709794153, 0.0011291977574560646,
                         0.0004754980986263272]

        # Recovery Rates:
        self.gamma_i_uv = [0.05454324458277713, 0.054999999999984624, 0.05493914662841438, 0.05499990759397898,
                           0.05499806747662576, 0.04100138873053479, 0.05477290137597402, 0.053825419002336575,
                           0.054529234511264964, 0.04000000000000034, 0.046392374251813036, 0.054999713367516335,
                           0.04000004218536563, 0.0400000026033707, 0.04012178719368275]

        self.gamma_i_fv = [0.05499214815619127, 0.051716264016610655, 0.05496900364583629, 0.05461994710842431,
                           0.05499999997555863, 0.04747274559193559, 0.05326993709415978, 0.054999468572432986,
                           0.05499999748127864, 0.045000001732971695, 0.052260588208621334, 0.054998278481102426,
                           0.045000030630856995, 0.05443487833553049, 0.05474162345518843]

        self.gamma_i_bv = [0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.047527421385129165, 0.047512182262614014,
                           0.04753550670362312, 0.047500000000000084, 0.06175487524141818, 0.06499802339400954,
                           0.047500619893717684, 0.04753664736862204, 0.06291341416198146]

        self.gamma_h_uv = [0.03467454100259981, 0.025387756477381746, 0.025004448376817784, 0.02761422403317452,
                           0.025000000000236458, 0.05251118105209822, 0.0549110466748263, 0.033067036304681066,
                           0.025000016934706253, 0.05499999999986076, 0.04197826269230298, 0.04834864909249086,
                           0.026320057827536415, 0.02500001559861162, 0.025028212484802824]

        self.gamma_h_fv = [0.030290563140145815, 0.030000000000000207, 0.03185761802081461, 0.04204778355488563,
                           0.05354273969484376, 0.054191972978106906, 0.04554706163804386, 0.030000011314920128,
                           0.03038156224719357, 0.05499999999999999, 0.04614880587376313, 0.04269476504178644,
                           0.03418548843830638, 0.030676406640121834, 0.03000935720574363]

        self.gamma_h_bv = [0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.0377777, 0.05024106727480235,
                           0.06122977561546748, 0.03339888791458119, 0.030000018417240464, 0.05951600677771558,
                           0.04358177542024513, 0.03491646714746562, 0.06491527721233392, 0.04218485117522834]

        # Death Rates:
        self.mu_i_uv = [0.0011898381320066536, 0.0016461696490861401, 5.557756084789282e-05, 0.0025929773893008544,
                        0.003332580613624076, 0.00038434227102280644, 0.0014475754401885475, 0.0006446404408645657,
                        0.00045090847272443433, 5.5555550000155024e-05, 0.0002842322533048449, 0.0033331446010170136,
                        0.0001346545061541397, 0.00016510386851820654, 0.001174139354820452]

        self.mu_i_fv = [0.000968991153713453, 0.0008031958184529309, 5.555555000134667e-06, 0.0007799791561391193,
                        0.00012356244610305732, 0.00028252658788236753, 0.0009695630490258248, 0.0007025163121404912,
                        0.0005389402123607347, 0.00022287710504662042, 0.0003852112799188121, 0.0028162301234761593,
                        1.658517276642031e-05, 0.00012902217569051903, 0.00020343028748589007]

        self.mu_i_bv = [5.555555000000009e-06, 5.555555000000009e-06, 5.555555000000009e-06, 5.555555000000009e-06,
                        5.555555000000009e-06, 5.555555000000009e-06, 0.00023056307212302528, 0.00016819448709593123,
                        3.36418613048688e-05, 0.0002482277445804881, 0.0003162945618154334, 0.0012679416366859486,
                        0.000820640351487956, 1.709677015416232e-05, 0.00017848713214707823]

        self.mu_h_uv = [0.0065387373871696065, 0.0027777723418094774, 0.0027777707610357793, 0.0027796898070118975,
                        0.002908950156740234, 0.002813558388157619, 0.007375748874639409, 0.0027833913951630404,
                        0.002781769011272498, 0.011425345371481198, 0.007260659510285804, 0.011103496703024114,
                        0.0027786870476711072, 0.0027777753029138794, 0.002778499268413278]

        self.mu_h_fv = [0.0007779467358999902, 0.0036787721921557924, 0.012782730610475765, 0.0007780821739797155,
                        0.010918170242898977, 0.01065527161079136, 0.013786555042614205, 0.0007777747122808551,
                        0.000782712451470584, 0.013888879999999729, 0.006634152071829002, 0.007622872559056293,
                        0.005436367402536328, 0.0014393769862645256, 0.0007832171167080535]

        self.mu_h_bv = [0.0008777700000000002, 0.0008777700000000002, 0.0008777700000000002, 0.0008777700000000002,
                        0.0008777700000000002, 0.0008777700000000002, 0.00978766881560744, 0.013888851250363472,
                        0.0073671130602453875, 0.01388887999999899, 0.013887747182280856, 0.01253248402718307,
                        0.0033045026940641932, 0.013875288010646254, 0.0020271548979584873]

        # Exposure to Susceptible and Recovered Rates:
        self.sigma_s_uv = [0.2797044028880083, 0.34431409936925694, 0.3580504182316429, 0.1166177252469825,
                           0.16997197499388433, 0.23405769641743218, 0.2769261712186397, 0.2566061899692816,
                           0.174547479075425, 0.0, 0.15547516429999247, 0.02161976137750099, 0.00810782879033839,
                           0.0018613327530079271, 2.7699159632632586e-10]

        self.sigma_s_fv = [0.2662392505973987, 0.33212344139038685, 0.3261725159472545, 0.1490687843082517,
                           0.17262913177692407, 0.2511983728144259, 0.29030016577761925, 0.26604931009425203,
                           0.19978065051003357, 0.019989520104101544, 0.29600857702146344, 0.09553721531017201,
                           0.03525179288803204, 0.013619697748734672, 0.2665718014138282]

        self.sigma_s_bv = [0.5, 0.5, 0.5, 0.5, 0.5, 0.14985202566588945, 0.21973900778289351, 0.2599201628582599,
                           0.20235521660244826, 0.021243110081864858, 0.3154426026890688, 0.07282721934828185,
                           0.06104703495295127, 0.014707636143268588, 0.23840873738908486]

        self.sigma_r_uv = [0.03953448795218578, 0.058203388178277304, 0.032928459326050485, 0.0515539619520361,
                           0.034443635234412906, 0.059315200907886834, 0.07371744803997626, 0.08680504616510387,
                           0.09303842995851996, 0.09084809679551148, 0.9042612659047722, 0.7424077412987622,
                           0.9062605142439981, 0.9961753414307617, 3.271895679946013e-07]

        self.sigma_r_fv = [0.019437664121463194, 0.021335162733075785, 0.029657269748876447, 0.011550470005043612,
                           0.022108016565385413, 0.026750229515834056, 0.027068151213753, 0.032421554164926925,
                           0.02804895096106269, 0.018719326095313793, 0.06399355798909162, 0.03261880318624616,
                           0.03236626547518978, 0.0027465977594754443, 0.011397488403402434]

        self.sigma_r_bv = [0.5, 0.5, 0.5, 0.5, 0.5, 0.36269708395031774, 0.0765039340194445, 0.04085645680385164,
                           0.0319032123657676, 0.01950260146638827, 0.0672133875063915, 0.05921676760645461,
                           0.00578248066924314, 0.0016780890594801368, 0.03637156804177161]

        # Infection Rates:
        self.zeta_s_uv = [0.0007318991017170318, 0.0011574000234550435, 0.0017036948546286818, 0.004362887453730158,
                          0.013457377607078725, 0.006346703985227398, 0.011831358692383566, 0.01252818896039103,
                          0.020312861456003757, 0.033087401587158706, 0.01400141647252155, 0.0008811232498273897,
                          0.049999988741923546, 0.04999999994431923, 6.647410305418711e-06]

        self.zeta_s_fv = [6.1106906552897415e-06, 0.00013962269248590849, 0.00027731392827476555,
                          0.0008033915116582933, 0.0013398147739391419, 0.0006858134315300495, 0.00034256391767068574,
                          0.0003035016049242131, 0.00046329586694962636, 0.0011518632138663876, 5.289629342576374e-07,
                          2.5765181231485192e-08, 0.00015346896176303293, 0.00031249562807744675, 1.929435442860061e-09]

        self.zeta_s_bv = [0.0003000000000000001, 0.0003000000000000001, 0.0003000000000000001, 0.0003000000000000001,
                          0.0003000000000000001, 0.001149491404118182, 0.004036702171532546, 3.397575846220136e-10,
                          0.0001682986024646966, 0.000999212015626945, 0.00019274732954734208, 5.880573818801605e-05,
                          0.0009881243669758186, 0.001658550667957768, 0.0005631708299375719]

        self.zeta_r_uv = [0.0024909333130088587, 0.0007187527866670652, 0.00023545233240417074, 0.0057906673455601744,
                          6.310338210774314e-07, 0.0011364601214976788, 0.00063194566537908, 1.2389820280844788e-10,
                          2.420593336882604e-09, 0.0033726612629533664, 8.856118519284806e-06, 5.118492442954259e-06,
                          0.04999942155546967, 0.049999999998551835, 1.127830620561987e-08]

        self.zeta_r_fv = [0.0006450657128552588, 1.2115125569422958e-13, 2.265172591711384e-06, 0.00021820919238626935,
                          1.872674237901606e-12, 4.2976049414082955e-06, 2.0901160795103735e-06, 6.318968785489765e-07,
                          1.0528939442533414e-10, 2.8798764222455142e-05, 0.0003786947524119257, 1.621949413471713e-08,
                          0.00022645875956498336, 0.000495940501522658, 0.000724824228763912]

        self.zeta_r_bv = [0.004, 0.004, 0.004, 0.004, 0.004, 0.006979005796484052, 0.0010701088453985836,
                          0.0013051640745155204, 0.0006450084026508098, 0.0010040242434811648, 0.0005112768916504632,
                          7.871312454313601e-05, 0.0009921237117573275, 0.0015537442266259076, 0.002874814750625074]

        # Hyperparameters for reward function.
        self.economic_and_social_rate_lower_limit = 70
        self.economic_and_social_rate_coefficient = 1
        self.infection_coefficient = 500_000
        self.penalty_coefficient = 1_000
        self.deceased_coefficient = 10_000

        self.max_timesteps = 181
        self.timestep = 0

        # To help avoid rapidly changing policies.
        self.action_history = []
        self.previous_action = 0
        self.current_action = 0

        self.allowed_actions = [True, True, True, True, True]
        self.required_actions = [False, False, False, False, False]
        self.allowed_actions_numbers = [1 for _ in range(self.action_space.n)]

        self.min_no_npm_pm_period = 14
        self.min_sdm_period = 28
        self.min_lockdown_period = 14
        self.min_mask_mandate_period = 28
        self.min_vaccination_mandate_period = 0

        self.max_no_npm_pm_period = 56
        self.max_sdm_period = 112
        self.max_lockdown_period = 42
        self.max_mask_mandate_period = 180
        self.max_vaccination_mandate_period = 0

        self.no_npm_pm_counter = 0
        self.sdm_counter = 0
        self.lockdown_counter = 0
        self.mask_mandate_counter = 0
        self.vaccination_mandate_counter = 0

        self.new_cases = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """This method resets the environment and returns the state as the observation.

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)"""

        # Population Dynamics by Epidemiological Compartments:
        self.number_of_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible'].iloc[0]
        self.number_of_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed'].iloc[0]
        self.number_of_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected'].iloc[0]
        self.number_of_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized'].iloc[0]
        self.number_of_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered'].iloc[0]
        self.number_of_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased'].iloc[0]

        # Population Dynamics by Vaccination Status:
        self.number_of_unvaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'unvaccinated_individuals'].iloc[0]
        self.number_of_fully_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'fully_vaccinated_individuals'].iloc[0]
        self.number_of_booster_vaccinated_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'boosted_individuals'].iloc[0]

        # Susceptible Compartment by Vaccination Status:
        self.number_of_unvaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_UV'].iloc[0]
        self.number_of_fully_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_FV'].iloc[0]
        self.number_of_booster_vaccinated_susceptible_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Susceptible_BV'].iloc[0]

        # Exposed Compartment by Vaccination Status:
        self.number_of_unvaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_UV'].iloc[0]
        self.number_of_fully_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_FV'].iloc[0]
        self.number_of_booster_vaccinated_exposed_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Exposed_BV'].iloc[0]

        # Infected Compartment by Vaccination Status:
        self.number_of_unvaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_UV'].iloc[0]
        self.number_of_fully_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_FV'].iloc[0]
        self.number_of_booster_vaccinated_infected_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Infected_BV'].iloc[0]

        # Hospitalized Compartment by Vaccination Status:
        self.number_of_unvaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_UV'].iloc[0]
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_FV'].iloc[0]
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Hospitalized_BV'].iloc[0]

        # Recovered Compartment by Vaccination Status:
        self.number_of_unvaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_UV'].iloc[0]
        self.number_of_fully_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_FV'].iloc[0]
        self.number_of_booster_vaccinated_recovered_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Recovered_BV'].iloc[0]

        # Deceased Compartment by Vaccination Status:
        self.number_of_unvaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_UV'].iloc[0]
        self.number_of_fully_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_FV'].iloc[0]
        self.number_of_booster_vaccinated_deceased_individuals = \
            self.covid_data.loc[self.covid_data['date'] == self.start_date, 'Deceased_BV'].iloc[0]

        # LISTS FOR PLOTTING:
        # Lists by Epidemiological Compartments:
        self.number_of_susceptible_individuals_list = [self.number_of_susceptible_individuals]
        self.number_of_exposed_individuals_list = [self.number_of_exposed_individuals]
        self.number_of_infected_individuals_list = [self.number_of_infected_individuals]
        self.number_of_hospitalized_individuals_list = [self.number_of_hospitalized_individuals]
        self.number_of_recovered_individuals_list = [self.number_of_recovered_individuals]
        self.number_of_deceased_individuals_list = [self.number_of_deceased_individuals]

        # Lists by Vaccination Status:
        self.number_of_unvaccinated_individuals_list = [self.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = [self.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = [self.number_of_booster_vaccinated_individuals]

        # Lists for Epidemiological Compartments by Vaccination Status:
        # Susceptible Compartment
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.number_of_booster_vaccinated_susceptible_individuals]

        # Exposed Compartment
        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.number_of_booster_vaccinated_exposed_individuals]

        # Infected Compartment
        self.number_of_unvaccinated_infected_individuals_list = \
            [self.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.number_of_booster_vaccinated_infected_individuals]

        # Hospitalized Compartment
        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.number_of_booster_vaccinated_hospitalized_individuals]

        # Recovered Compartment
        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.number_of_booster_vaccinated_recovered_individuals]

        # Deceased Compartment
        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.number_of_booster_vaccinated_deceased_individuals]

        self.economic_and_public_perception_rate = 100
        self.economic_and_public_perception_rate_list = [self.economic_and_public_perception_rate]

        self.new_cases = []

        self.timestep = 0

        """To help avoid rapidly changing policies."""
        self.action_history = []
        self.previous_action = 0
        self.current_action = 0

        # Counter to keep track of the consecutive times an action was taken.
        self.no_npm_pm_counter = 0
        self.sdm_counter = 0
        self.lockdown_counter = 0
        self.mask_mandate_counter = 0
        self.vaccination_mandate_counter = 0

        # Boolean to check whether an action is allowed.
        no_npm_pm_allowed = True
        sdm_allowed = True
        lockdown_allowed = True
        mask_mandate_allowed = True
        vaccination_mandate_allowed = True

        no_npm_pm_required, sdm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required = \
            False, False, False, False, False

        self.allowed_actions = [no_npm_pm_allowed, sdm_allowed, lockdown_allowed,
                                mask_mandate_allowed, vaccination_mandate_allowed]
        self.required_actions = \
            [no_npm_pm_required, sdm_required, lockdown_required, mask_mandate_required, vaccination_mandate_required]
        self.allowed_actions_numbers = [1 for _ in range(self.action_space.n)]

        # Simpler observation:
        observation = \
            [self.number_of_infected_individuals / self.population,
             self.economic_and_public_perception_rate / 100, self.previous_action, self.current_action]

        info = {}

        return observation, info

    def step(self, action):
        """This method implements what happens when the agent takes a particular action. It changes the rate at which
        new people are infected, defines the rewards for the various states, and determines when the episode ends.

        :param action: - Integer in the range 0 to 1 inclusive.

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)
                 reward: - (Float value that's used to measure the performance of the agent.)
                 done: - (Boolean describing whether the episode has ended.)
                 info: - (A dictionary that can be used to provide additional implementation information.)"""

        self.action_history.append(action)

        if len(self.action_history) == 1:
            self.previous_action = 0
        else:
            self.previous_action = self.action_history[-2]
        self.current_action = action

        # This index helps to use the different parameter values for the different splits.
        # AD: Converts range(n) into [7 (10x), 8 (28x), 9 (28x), 10 (28x), ...]
        #     28 = 4 weeks. 214 = start date (october)
        index = int(np.floor((self.timestep + 214) / 28))

        # Updating the action dependent parameters:
        # Switch from a discrete action space to a multi-discrete action space.
        if action == 0:  # No NPM or PM taken. 7.3
            self.beta = self.beta_original[index] * 1.4 \
                if self.number_of_infected_individuals / self.population >= 0.001 \
                else self.beta_original[index] * 1.1
            self.economic_and_public_perception_rate = min(1.005 * self.economic_and_public_perception_rate, 100) \
                if self.number_of_infected_individuals / self.population < 0.001 \
                else 0.999 * self.economic_and_public_perception_rate
            self.no_npm_pm_counter += 1
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 1:  # SDM
            self.beta = self.beta_original[index] * 0.95
            self.economic_and_public_perception_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 2:  # Lockdown (Closure of non-essential business, schools, gyms...) 0.997
            self.beta = self.beta_original[index] * 0.85
            self.economic_and_public_perception_rate *= 0.997
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter = 0
        elif action == 3:  # Public Mask Mandates 0.9975
            self.beta = self.beta_original[index] * 0.925
            self.economic_and_public_perception_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 4:  # Vaccination Mandates 0.994
            self.beta = self.beta_original[index] * 0.95
            self.economic_and_public_perception_rate *= 0.994
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 5:  # SDM and Public Mask Mandates 0.9965
            self.beta = self.beta_original[index] * 0.875
            self.economic_and_public_perception_rate *= 0.9965
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 6:  # SDM and Vaccination Mandates 0.993
            self.beta = self.beta_original[index] * 0.825
            self.economic_and_public_perception_rate *= 0.993
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 7:  # Lockdown and Public Mask Mandates 0.9965
            self.beta = self.beta_original[index] * 0.75
            self.economic_and_public_perception_rate *= 0.994
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter = 0
        elif action == 8:  # Lockdown and Vaccination Mandates 0.993
            self.beta = self.beta_original[index] * 0.80
            self.economic_and_public_perception_rate *= 0.993
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter = 0
            self.vaccination_mandate_counter += 1

        elif action == 9:  # Public Mask Mandates and Vaccination Mandates 0.9935
            self.beta = self.beta_original[index] * 0.90
            self.economic_and_public_perception_rate *= 0.9935
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1
        elif action == 10:  # SDM, Public Mask Mandates and Vaccination Mandates 0.9925
            self.beta = self.beta_original[index] * 0.60
            self.economic_and_public_perception_rate *= 0.9925
            self.no_npm_pm_counter = 0
            self.sdm_counter += 1
            self.lockdown_counter = 0
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1

        elif action == 11:  # Lockdown, Public Mask Mandates and Vaccination Mandates 0.9925
            self.beta = self.beta_original[index] * 0.60
            self.economic_and_public_perception_rate *= 0.9925
            self.no_npm_pm_counter = 0
            self.sdm_counter = 0
            self.lockdown_counter += 1
            self.mask_mandate_counter += 1
            self.vaccination_mandate_counter += 1

        self.compute_population_dynamics(action)
        self.economic_and_public_perception_rate_list.append(self.economic_and_public_perception_rate)

        # Checking which actions are allowed:
        # Potential Violations (If the action is not taken in the next time-step.):
        no_npm_pm_min_period_violation = True if (0 < self.no_npm_pm_counter < self.min_no_npm_pm_period) else False
        sdm_min_period_violation = True if (0 < self.sdm_counter < self.min_sdm_period) else False
        lockdown_min_period_violation = True if (0 < self.lockdown_counter < self.min_lockdown_period) else False
        mask_mandate_min_period_violation = \
            True if (0 < self.mask_mandate_counter < self.min_mask_mandate_period) else False
        vaccination_mandate_min_period_violation = \
            True if (0 < self.vaccination_mandate_counter < self.min_vaccination_mandate_period) else False

        # Potential Violations (If the action is taken in the next time-step.):
        no_npm_pm_max_period_violation = True if (self.no_npm_pm_counter >= self.max_no_npm_pm_period) else False
        sdm_max_period_violation = True if (self.sdm_counter >= self.max_sdm_period) else False
        lockdown_max_period_violation = True if (self.lockdown_counter >= self.max_lockdown_period) else False
        mask_mandate_max_period_violation = True if \
            (self.mask_mandate_counter >= self.max_mask_mandate_period) else False
        vaccination_mandate_max_period_violation = \
            True if (self.vaccination_mandate_counter >= self.max_vaccination_mandate_period) else False

        # Required Actions (As in not taking them will result in minimum violation):
        no_npm_pm_required = True if no_npm_pm_min_period_violation else False
        sdm_required = True if sdm_min_period_violation else False
        lockdown_required = True if lockdown_min_period_violation else False
        mask_mandate_required = True if mask_mandate_min_period_violation else False
        vaccination_mandate_required = True if vaccination_mandate_min_period_violation else False

        # Allowed Actions
        no_npm_pm_allowed = \
            True if ((not sdm_min_period_violation)
                     and (not lockdown_min_period_violation)
                     and (not mask_mandate_min_period_violation)
                     and (not vaccination_mandate_min_period_violation)
                     and (not no_npm_pm_max_period_violation)) else False

        sdm_allowed = True if ((not no_npm_pm_min_period_violation)
                               and (not lockdown_min_period_violation)
                               and (not sdm_max_period_violation)) else False

        lockdown_allowed = True if ((not no_npm_pm_min_period_violation)
                                    and (not sdm_min_period_violation)
                                    and (not lockdown_max_period_violation)) else False

        mask_mandate_allowed = True if ((not no_npm_pm_min_period_violation)
                                        and (not mask_mandate_max_period_violation)) else False

        vaccination_mandate_allowed = True if ((not no_npm_pm_min_period_violation)
                                               and (not vaccination_mandate_max_period_violation)) else False

        # Updating the lists for allowed and required actions.
        self.allowed_actions = [no_npm_pm_allowed, sdm_allowed, lockdown_allowed, mask_mandate_allowed,
                                vaccination_mandate_allowed]
        self.required_actions = [no_npm_pm_required, sdm_required, lockdown_required,
                                 mask_mandate_required, vaccination_mandate_required]

        # Logic to determine which action as per the numbers is allowed.
        # (Each list within the list is a set of actions corresponding to the five "isolated" actions.)
        action_association_list = [[0], [1, 5, 6, 10], [2, 7, 8, 11], [3, 5, 7, 9, 10, 11], [4, 6, 8, 9, 10, 11]]
        actions_allowed = None

        # First we simply go through the required actions and find the set of associated actions. This can lead to a
        # situation in which for e.g., the mask mandate action is required but not all other actions in the action
        # association list such as lockdown are allowed are included. We remove them with the next for loop.
        for i in range(5):
            if self.required_actions[i]:
                if actions_allowed is None:
                    actions_allowed = set(action_association_list[i])
                else:
                    # Set intersection operator.
                    actions_allowed = actions_allowed & set(action_association_list[i])

        # Here we check if the "actions_allowed" set contains any actions that are in fact not allowed
        # (and not required). We remove such actions from the set with by taking a difference between the sets.
        for i in range(5):
            if not self.allowed_actions[i] and not self.required_actions[i]:
                if actions_allowed is None:
                    break
                else:
                    actions_allowed = actions_allowed.difference(set(action_association_list[i]))

        # Exception case.
        if actions_allowed is None:
            for i in range(5):
                if self.allowed_actions[i]:
                    if actions_allowed is None:
                        actions_allowed = set(action_association_list[i])
                    else:
                        actions_allowed = actions_allowed.union(set(action_association_list[i]))
            for i in range(5):
                if not self.allowed_actions[i]:
                    actions_allowed = actions_allowed.difference(set(action_association_list[i]))

        actions_allowed = list(actions_allowed)
        self.allowed_actions_numbers = [1 if i in actions_allowed else 0 for i in range(self.action_space.n)]

        # Reward
        reward = ((-self.infection_coefficient * self.number_of_infected_individuals / self.population)
                  + self.economic_and_public_perception_rate)

        self.timestep += 1

        # Simplified observation:
        observation = \
            [self.number_of_infected_individuals / self.population,
             self.economic_and_public_perception_rate / 100, self.previous_action, self.current_action]

        # The episode terminates when the number of infected people becomes greater than 25 % of the population.
        terminated = True if (self.number_of_infected_individuals >= 0.99 * self.population or
                        self.timestep >= self.max_timesteps) else False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def compute_population_dynamics(self, action):
        """This method computes the action dependent population dynamics
        :parameter action: Integer - Represents the action taken by the agent."""

        # Action dependent vaccination rates.
        if action in [3, 5, 6, 7]:
            percentage_unvaccinated_to_fully_vaccinated = 0.007084760245099044
            # percentage_fully_vaccinated_to_booster_vaccinated = 0.0017285714029114
            percentage_fully_vaccinated_to_booster_vaccinated = \
                self.covid_data['percentage_fully_vaccinated_to_boosted'].iloc[self.timestep + 214]
        else:
            percentage_unvaccinated_to_fully_vaccinated = \
                self.covid_data['percentage_unvaccinated_to_fully_vaccinated'].iloc[self.timestep + 214]
            percentage_fully_vaccinated_to_booster_vaccinated = \
                self.covid_data['percentage_fully_vaccinated_to_boosted'].iloc[self.timestep + 214]

        # Index to use the different model parameter values for the different splits.
        index = int(np.floor((self.timestep + 214) / 28))

        # model_parameters = [
        #     self.beta, self.alpha,
        #     self.sigma_s_uv, self.sigma_s_fv, self.sigma_s_bv, self.sigma_r_uv, self.sigma_r_fv, self.sigma_r_bv,
        #     self.zeta_s_uv, self.zeta_s_fv, self.zeta_s_bv, self.zeta_r_uv, self.zeta_r_fv, self.zeta_r_bv,
        #     self.gamma_i_uv, self.gamma_i_fv, self.gamma_i_bv, self.gamma_h_uv, self.gamma_h_fv, self.gamma_h_bv,
        #     self.delta_uv, self.delta_fv, self.delta_bv,
        #     self.mu_i_uv, self.mu_i_fv, self.mu_i_bv, self.mu_h_uv, self.mu_h_fv, self.mu_h_bv
        #     ]
        #
        # standard_deviation = 0.05
        # # for model_parameter in model_parameters:

        val = 0.05  # AD: Why?
        mu, sigma = self.beta, val * self.beta
        # AD: Why brownian motion?
        self.beta = np.random.normal(mu, sigma, 1)

        mu, sigma = self.alpha[index], val * self.alpha[index]
        alpha = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_uv[index], val * self.delta_uv[index]
        delta_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_fv[index], val * self.delta_fv[index]
        delta_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.delta_bv[index], val * self.delta_bv[index]
        delta_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_uv[index], val * self.gamma_i_uv[index]
        gamma_i_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_fv[index], val * self.gamma_i_fv[index]
        gamma_i_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_i_bv[index], val * self.gamma_i_bv[index]
        gamma_i_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_uv[index], val * self.gamma_h_uv[index]
        gamma_h_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_fv[index], val * self.gamma_h_fv[index]
        gamma_h_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.gamma_h_bv[index], val * self.gamma_h_bv[index]
        gamma_h_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_uv[index], val * self.mu_i_uv[index]
        mu_i_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_fv[index], val * self.mu_i_fv[index]
        mu_i_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_i_bv[index], val * self.mu_i_bv[index]
        mu_i_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_uv[index], val * self.mu_h_uv[index]
        mu_h_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_fv[index], val * self.mu_h_fv[index]
        mu_h_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.mu_h_bv[index], val * self.mu_h_bv[index]
        mu_h_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_uv[index], val * self.sigma_s_uv[index]
        sigma_s_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_fv[index], val * self.sigma_s_fv[index]
        sigma_s_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_s_bv[index], val * self.sigma_s_bv[index]
        sigma_s_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_uv[index], val * self.sigma_r_uv[index]
        sigma_r_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_fv[index], val * self.sigma_r_fv[index]
        sigma_r_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.sigma_r_bv[index], val * self.sigma_r_bv[index]
        sigma_r_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_uv[index], val * self.zeta_s_uv[index]
        zeta_s_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_fv[index], val * self.zeta_s_fv[index]
        zeta_s_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_s_bv[index], val * self.zeta_s_bv[index]
        zeta_s_bv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_uv[index], val * self.zeta_r_uv[index]
        zeta_r_uv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_fv[index], val * self.zeta_r_fv[index]
        zeta_r_fv = np.random.normal(mu, sigma, 1)

        mu, sigma = self.zeta_r_bv[index], val * self.zeta_r_bv[index]
        zeta_r_bv = np.random.normal(mu, sigma, 1)

        # Susceptible Compartment
        number_of_unvaccinated_susceptible_individuals = \
            int(self.number_of_unvaccinated_susceptible_individuals
                - (self.beta * self.number_of_unvaccinated_susceptible_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_s_uv * self.number_of_unvaccinated_exposed_individuals -
                percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_susceptible_individuals)

        number_of_fully_vaccinated_susceptible_individuals = \
            int(self.number_of_fully_vaccinated_susceptible_individuals
                - self.beta * self.number_of_fully_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + sigma_s_fv * self.number_of_fully_vaccinated_exposed_individuals +
                percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_susceptible_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_susceptible_individuals)

        number_of_booster_vaccinated_susceptible_individuals = \
            int(self.number_of_booster_vaccinated_susceptible_individuals
                - self.beta * self.number_of_booster_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + sigma_s_bv * self.number_of_booster_vaccinated_exposed_individuals +
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_susceptible_individuals)

        number_of_susceptible_individuals = \
            number_of_unvaccinated_susceptible_individuals + \
            number_of_fully_vaccinated_susceptible_individuals + \
            number_of_booster_vaccinated_susceptible_individuals

        self.number_of_unvaccinated_susceptible_individuals_list.append(
            number_of_unvaccinated_susceptible_individuals)
        self.number_of_fully_vaccinated_susceptible_individuals_list.append(
            number_of_fully_vaccinated_susceptible_individuals)
        self.number_of_booster_vaccinated_susceptible_individuals_list.append(
            number_of_booster_vaccinated_susceptible_individuals)
        self.number_of_susceptible_individuals_list.append(number_of_susceptible_individuals)

        # Exposed Compartment
        number_of_unvaccinated_exposed_individuals = \
            int(self.number_of_unvaccinated_exposed_individuals
                + self.beta * self.number_of_unvaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_unvaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
                - zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
                - sigma_s_uv * self.number_of_unvaccinated_exposed_individuals
                - sigma_r_uv * self.number_of_unvaccinated_exposed_individuals
                - percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_exposed_individuals)

        number_of_fully_vaccinated_exposed_individuals = \
            int(self.number_of_fully_vaccinated_exposed_individuals
                + self.beta * self.number_of_fully_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_fully_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                - zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                - sigma_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                - sigma_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                + percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_exposed_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_exposed_individuals)

        number_of_booster_vaccinated_exposed_individuals = \
            int(self.number_of_booster_vaccinated_exposed_individuals
                + self.beta * self.number_of_booster_vaccinated_susceptible_individuals *
                (self.number_of_infected_individuals ** alpha) / self.population
                + (self.beta * self.number_of_booster_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                - zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                - zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                - sigma_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                - sigma_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                + percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_exposed_individuals)

        number_of_exposed_individuals = \
            number_of_unvaccinated_exposed_individuals + \
            number_of_fully_vaccinated_exposed_individuals + \
            number_of_booster_vaccinated_exposed_individuals

        self.number_of_unvaccinated_exposed_individuals_list.append(
            number_of_unvaccinated_exposed_individuals)
        self.number_of_fully_vaccinated_exposed_individuals_list.append(
            number_of_fully_vaccinated_exposed_individuals)
        self.number_of_booster_vaccinated_exposed_individuals_list.append(
            number_of_booster_vaccinated_exposed_individuals)
        self.number_of_exposed_individuals_list.append(number_of_exposed_individuals)

        # Infected Compartment
        number_of_unvaccinated_infected_individuals = \
            int(self.number_of_unvaccinated_infected_individuals +
                zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
                + zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
                - delta_uv * self.number_of_unvaccinated_infected_individuals -
                gamma_i_uv * self.number_of_unvaccinated_infected_individuals -
                mu_i_uv * self.number_of_unvaccinated_infected_individuals)

        number_of_fully_vaccinated_infected_individuals = \
            int(self.number_of_fully_vaccinated_infected_individuals +
                zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
                + zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                - delta_fv * self.number_of_fully_vaccinated_infected_individuals -
                gamma_i_fv * self.number_of_fully_vaccinated_infected_individuals -
                mu_i_fv * self.number_of_fully_vaccinated_infected_individuals)

        number_of_booster_vaccinated_infected_individuals = \
            int(self.number_of_booster_vaccinated_infected_individuals +
                zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
                + zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                - delta_bv * self.number_of_booster_vaccinated_infected_individuals -
                gamma_i_bv * self.number_of_booster_vaccinated_infected_individuals -
                mu_i_bv * self.number_of_booster_vaccinated_infected_individuals)

        self.new_cases.append(int(
            zeta_s_uv * self.number_of_unvaccinated_exposed_individuals
            + zeta_r_uv * self.number_of_unvaccinated_exposed_individuals
            + zeta_s_fv * self.number_of_fully_vaccinated_exposed_individuals
            + zeta_r_fv * self.number_of_fully_vaccinated_exposed_individuals
            + zeta_s_bv * self.number_of_booster_vaccinated_exposed_individuals
            + zeta_r_bv * self.number_of_booster_vaccinated_exposed_individuals))

        number_of_infected_individuals = \
            number_of_unvaccinated_infected_individuals + \
            number_of_fully_vaccinated_infected_individuals + \
            number_of_booster_vaccinated_infected_individuals

        self.number_of_unvaccinated_infected_individuals_list.append(
            number_of_unvaccinated_infected_individuals)
        self.number_of_fully_vaccinated_infected_individuals_list.append(
            number_of_fully_vaccinated_infected_individuals)
        self.number_of_booster_vaccinated_infected_individuals_list.append(
            number_of_booster_vaccinated_infected_individuals)
        self.number_of_infected_individuals_list.append(number_of_infected_individuals)

        # Hospitalized Compartment
        number_of_unvaccinated_hospitalized_individuals = \
            int(self.number_of_unvaccinated_hospitalized_individuals +
                delta_uv * self.number_of_unvaccinated_infected_individuals -
                gamma_h_uv * self.number_of_unvaccinated_hospitalized_individuals -
                mu_h_uv * self.number_of_unvaccinated_hospitalized_individuals)

        number_of_fully_vaccinated_hospitalized_individuals = \
            int(self.number_of_fully_vaccinated_hospitalized_individuals +
                delta_fv * self.number_of_fully_vaccinated_infected_individuals -
                gamma_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals -
                mu_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals)

        number_of_booster_vaccinated_hospitalized_individuals = \
            int(self.number_of_booster_vaccinated_hospitalized_individuals +
                delta_bv * self.number_of_booster_vaccinated_infected_individuals -
                gamma_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals -
                mu_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals)

        number_of_hospitalized_individuals = \
            number_of_unvaccinated_hospitalized_individuals + \
            number_of_fully_vaccinated_hospitalized_individuals + \
            number_of_booster_vaccinated_hospitalized_individuals

        self.number_of_unvaccinated_hospitalized_individuals_list.append(
            number_of_unvaccinated_hospitalized_individuals)
        self.number_of_fully_vaccinated_hospitalized_individuals_list.append(
            number_of_fully_vaccinated_hospitalized_individuals)
        self.number_of_booster_vaccinated_hospitalized_individuals_list.append(
            number_of_booster_vaccinated_hospitalized_individuals)
        self.number_of_hospitalized_individuals_list.append(number_of_hospitalized_individuals)

        # Recovered Compartment
        number_of_unvaccinated_recovered_individuals = \
            int(self.number_of_unvaccinated_recovered_individuals
                - (self.beta * self.number_of_unvaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_uv * self.number_of_unvaccinated_exposed_individuals
                + gamma_i_uv * self.number_of_unvaccinated_infected_individuals
                + gamma_h_uv * self.number_of_unvaccinated_hospitalized_individuals
                - percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_recovered_individuals)

        number_of_fully_vaccinated_recovered_individuals = \
            int(self.number_of_fully_vaccinated_recovered_individuals
                - (self.beta * self.number_of_fully_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_fv * self.number_of_fully_vaccinated_exposed_individuals
                + gamma_i_fv * self.number_of_fully_vaccinated_infected_individuals
                + gamma_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals
                + percentage_unvaccinated_to_fully_vaccinated *
                self.number_of_unvaccinated_recovered_individuals -
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_recovered_individuals)

        number_of_booster_vaccinated_recovered_individuals = \
            int(self.number_of_booster_vaccinated_recovered_individuals
                - (self.beta * self.number_of_booster_vaccinated_recovered_individuals *
                   (self.number_of_infected_individuals ** alpha) / self.population)
                + sigma_r_bv * self.number_of_booster_vaccinated_exposed_individuals
                + gamma_i_bv * self.number_of_booster_vaccinated_infected_individuals +
                gamma_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals +
                percentage_fully_vaccinated_to_booster_vaccinated *
                self.number_of_fully_vaccinated_recovered_individuals)

        number_of_recovered_individuals = \
            number_of_unvaccinated_recovered_individuals + \
            number_of_fully_vaccinated_recovered_individuals + \
            number_of_booster_vaccinated_recovered_individuals

        self.number_of_unvaccinated_recovered_individuals_list.append(
            number_of_unvaccinated_recovered_individuals)
        self.number_of_fully_vaccinated_recovered_individuals_list.append(
            number_of_fully_vaccinated_recovered_individuals)
        self.number_of_booster_vaccinated_recovered_individuals_list.append(
            number_of_booster_vaccinated_recovered_individuals)
        self.number_of_recovered_individuals_list.append(number_of_recovered_individuals)

        # Deceased Compartment
        number_of_unvaccinated_deceased_individuals = \
            int(self.number_of_unvaccinated_deceased_individuals +
                mu_i_uv * self.number_of_unvaccinated_infected_individuals +
                mu_h_uv * self.number_of_unvaccinated_hospitalized_individuals)

        number_of_fully_vaccinated_deceased_individuals = \
            int(self.number_of_fully_vaccinated_deceased_individuals +
                mu_i_fv * self.number_of_fully_vaccinated_infected_individuals +
                mu_h_fv * self.number_of_fully_vaccinated_hospitalized_individuals)

        number_of_booster_vaccinated_deceased_individuals = \
            int(self.number_of_booster_vaccinated_deceased_individuals +
                mu_i_bv * self.number_of_booster_vaccinated_infected_individuals +
                mu_h_bv * self.number_of_booster_vaccinated_hospitalized_individuals)

        number_of_deceased_individuals = \
            number_of_unvaccinated_deceased_individuals + \
            number_of_fully_vaccinated_deceased_individuals + \
            number_of_booster_vaccinated_deceased_individuals

        self.number_of_unvaccinated_deceased_individuals_list.append(
            number_of_unvaccinated_deceased_individuals)
        self.number_of_fully_vaccinated_deceased_individuals_list.append(
            number_of_fully_vaccinated_deceased_individuals)
        self.number_of_booster_vaccinated_deceased_individuals_list.append(
            number_of_booster_vaccinated_deceased_individuals)
        self.number_of_deceased_individuals_list.append(number_of_deceased_individuals)

        # Population Dynamics by Vaccination Status
        self.number_of_unvaccinated_individuals = \
            self.number_of_unvaccinated_individuals \
            - percentage_unvaccinated_to_fully_vaccinated * self.number_of_unvaccinated_individuals

        self.number_of_fully_vaccinated_individuals = \
            self.number_of_fully_vaccinated_individuals \
            + percentage_unvaccinated_to_fully_vaccinated * self.number_of_unvaccinated_individuals \
            - percentage_fully_vaccinated_to_booster_vaccinated * self.number_of_fully_vaccinated_individuals

        self.number_of_booster_vaccinated_individuals = \
            self.number_of_booster_vaccinated_individuals \
            + percentage_fully_vaccinated_to_booster_vaccinated * self.number_of_fully_vaccinated_individuals

        self.number_of_unvaccinated_individuals_list.append(self.number_of_unvaccinated_individuals)
        self.number_of_fully_vaccinated_individuals_list.append(self.number_of_fully_vaccinated_individuals)
        self.number_of_booster_vaccinated_individuals_list.append(self.number_of_booster_vaccinated_individuals)

        # Synchronizing the global variables with the updated local variables:
        self.number_of_unvaccinated_susceptible_individuals = \
            number_of_unvaccinated_susceptible_individuals
        self.number_of_fully_vaccinated_susceptible_individuals = \
            number_of_fully_vaccinated_susceptible_individuals
        self.number_of_booster_vaccinated_susceptible_individuals = \
            number_of_booster_vaccinated_susceptible_individuals
        self.number_of_susceptible_individuals = number_of_susceptible_individuals

        self.number_of_unvaccinated_exposed_individuals = \
            number_of_unvaccinated_exposed_individuals
        self.number_of_fully_vaccinated_exposed_individuals = \
            number_of_fully_vaccinated_exposed_individuals
        self.number_of_booster_vaccinated_exposed_individuals = \
            number_of_booster_vaccinated_exposed_individuals
        self.number_of_exposed_individuals = number_of_exposed_individuals

        self.number_of_unvaccinated_infected_individuals = \
            number_of_unvaccinated_infected_individuals
        self.number_of_fully_vaccinated_infected_individuals = \
            number_of_fully_vaccinated_infected_individuals
        self.number_of_booster_vaccinated_infected_individuals = \
            number_of_booster_vaccinated_infected_individuals
        self.number_of_infected_individuals = number_of_infected_individuals

        self.number_of_unvaccinated_hospitalized_individuals = \
            number_of_unvaccinated_hospitalized_individuals
        self.number_of_fully_vaccinated_hospitalized_individuals = \
            number_of_fully_vaccinated_hospitalized_individuals
        self.number_of_booster_vaccinated_hospitalized_individuals = \
            number_of_booster_vaccinated_hospitalized_individuals
        self.number_of_hospitalized_individuals = number_of_hospitalized_individuals

        self.number_of_unvaccinated_recovered_individuals = \
            number_of_unvaccinated_recovered_individuals
        self.number_of_fully_vaccinated_recovered_individuals = \
            number_of_fully_vaccinated_recovered_individuals
        self.number_of_booster_vaccinated_recovered_individuals = \
            number_of_booster_vaccinated_recovered_individuals
        self.number_of_recovered_individuals = number_of_recovered_individuals

        self.number_of_unvaccinated_deceased_individuals = \
            number_of_unvaccinated_deceased_individuals
        self.number_of_fully_vaccinated_deceased_individuals = \
            number_of_fully_vaccinated_deceased_individuals
        self.number_of_booster_vaccinated_deceased_individuals = \
            number_of_booster_vaccinated_deceased_individuals
        self.number_of_deceased_individuals = number_of_deceased_individuals

    def render(self, mode='human'):
        """This method renders the statistical graph of the population.

        :param mode: 'human' renders to the current display or terminal and returns nothing."""

        return
