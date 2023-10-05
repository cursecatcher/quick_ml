#!/usr/bin/env python3 

import argparse
import csv, json
import stat, sys
from typing import Dict, Iterable, List 
import os, shutil, getpass, enum, subprocess, time 



class ArgparseParameterNames( enum.Enum ):
    # I/O & GENERIC STUFF
    INPUT_DATASET = "input_data"
    OUTPUT_FOLDER = "outfolder"
    TEST_SETS = "test_sets"
    FEATURE_SETS = "feature_lists"
    RULES_SETS = "rules"
    ESTIMATORS = "estimators"
    TRAINED_MODELS = "load_from"

    TSET_PROPORTION = "tsize"
    TARGET_FEATURE = "target"
    TARGET_POS_LABELS = "pos_labels"
    TARGET_NEG_LABELS = "neg_labels"

    NUM_PROCESSES = "np"
    
    # EVALUATION
    EV_LOO = "loo"
    EV_NCV = "ncv"
    EV_NTRIALS = "trials"
    # FEATURE SELECTION
    FS_MIN_NF = "min_nf"
    FS_MAX_NF = "max_nf"
    FS_MIN_AUC = "min_auc"
    FS_NTEST_EVAL = "fs_ntrials"
    FS_NCV_EVAL = "fs_ncv"
    # EXPLANATION 
    EX_RULES_FILES = "rules"
    EX_NCLUSTERS = "clusters"
    EX_MAX_NRULES = "max_rules"
    EX_MIN_NRULES = "min_rules"
    EX_NTEST_EVAL = FS_NTEST_EVAL
    EX_NCV_EVAL = FS_NCV_EVAL
    EX_LOO = EV_LOO
    # GENETIC SELECTION
    GA_ENABLE_SELECTION = "enable_ga"
    GA_EDGE_THRESHOLD = "edge_threshold"
    GA_NRUNS = "nga"
    GA_NGEN = "ngen"
    GA_POPSIZE = "popsize"
    GA_MAX_NF = FS_MAX_NF
    FLAG_DISABLE_EVAL = "no_eval"

    GA_PROB_MUTATION = "mprob"
    GA_PROB_CROSSOVER = "cprob"
    # PLOT CORRELATION GRAPHS 
    PG_SUBGRAPH_ONLY = "subgraph_only"

    GRAPH_VW_SCORES = "vws"
    GRAPH_EW_SCORE = "ew"


    # TUNING
    TU_TRAINED_MODELS = TRAINED_MODELS
    TU_EXHAUSTIVE_SEARCH = "exhaustive"


    @classmethod
    def get_argvalue(cls, d: Dict, k):
        """ """
        return d.get( k.value )

    @classmethod
    def build_dict_args(cls, d: Dict, keys: List) -> Dict:
        return { k.value: cls.get_argvalue( d, k ) for k in keys }

    @classmethod
    def build_list_args(cls, d: Dict, keys: List) -> List:
        return [ cls.get_argvalue( d, k ) for k in keys ]

    @classmethod
    def replace_keys(cls, d: Dict) -> Dict:
        return { k.value: v for k, v in d.items() }



class SupportedScores( enum.Enum ):
    ANOVA = "anova"
    PEARSON_R = "pearson"
    MUTUAL_INFO = "MI"
    MCC = "mcc"
    T_TEST = "t_test"
    LOGISTIC = "logistic"
    POINT_BISERIAL = "pbc"
    # CHI_SQUARED = "chi2"


    @classmethod
    def get_scores(cls, selected_scores: Iterable[str]):
        """ ... 
        """
        the_map = { sc.value: sc for sc in cls }
        obtained = { name: the_map.get( name ) for name in selected_scores }
        if all( obtained.values() ):
            return list( obtained.values() )
        else:
            unknown = [ name for name, instance in obtained.items() if not instance ]
            print(f"Unrecognized scores: {unknown}")
            return list( filter( lambda item: item, obtained.values() ) )




class SupportedEdgeScorer( enum.Enum ):
    SPEARMAN_R = "spearman"
    PEARSON_R = SupportedScores.PEARSON_R.value



def get_opt( item: ArgparseParameterNames ):
    return f"--{item.value}"


class Parser:
    """ Auxiliary methods to parse command line arguments """


    @classmethod
    def set_evaluation_parameters(cls, parser: argparse.ArgumentParser):
        # cls.get_standard_parser( parser )
        parser.add_argument( get_opt( ArgparseParameterNames.EV_LOO ), action="store_true", help="Enable leave-one-out cross-validation")

    @classmethod
    def set_fselection_parameters(cls, parser: argparse.ArgumentParser):
        # cls.get_standard_parser( parser ) 
        parser.add_argument( get_opt( ArgparseParameterNames.FS_MIN_NF ), type=int, default=1, help="Minimum number of features to be selected")
        parser.add_argument( get_opt( ArgparseParameterNames.FS_MAX_NF ), type=int, help="Maximum number of features to be selected")
        parser.add_argument( get_opt( ArgparseParameterNames.FS_MIN_AUC ), type=float, required=False, default=0.8, help="Minimum AUC to be considered")
        
        parser.add_argument( get_opt( ArgparseParameterNames.FS_NTEST_EVAL ), type=int, default=3, help="Number of trials during feature evaluation phase")
        parser.add_argument( get_opt( ArgparseParameterNames.FS_NCV_EVAL ), type=int, default=10, help="Number of cross-validation folds during feature evaluation phase")

        cls.set_evaluation_parameters( parser )


    @classmethod
    def set_explaination_parameters(cls, parser: argparse.ArgumentParser):
        # cls.get_standard_parser( parser )
        parser.add_argument("-r", get_opt( ArgparseParameterNames.EX_RULES_FILES ), type=str, nargs="*", default=list())
        parser.add_argument("-c", get_opt( ArgparseParameterNames.EX_NCLUSTERS ), type=int, default=3)
        parser.add_argument( get_opt( ArgparseParameterNames.EX_NTEST_EVAL ), type=int, default=3)
        parser.add_argument( get_opt( ArgparseParameterNames.EX_NCV_EVAL ), type=int, default=10)
        # parser.add_argument( get_opt( ArgparseParameterNames.EX_LOO ), action="store_true")
        parser.add_argument( get_opt( ArgparseParameterNames.EX_MAX_NRULES ), type=int, required=False, default=20)
        parser.add_argument( get_opt( ArgparseParameterNames.EX_MIN_NRULES ), type=int, required=False, default=2)

        parser.add_argument( get_opt( ArgparseParameterNames.GA_ENABLE_SELECTION ), action="store_true" )
        cls.set_ga_parameters( parser )
        

    @classmethod
    def set_ga_parameters(cls, parser: argparse.ArgumentParser):
        # cls.get_standard_parser( parser )
        # parser.add_argument( get_opt( ArgparseParameterNames.GA_EDGE_THRESHOLD ), type=float, default=0.5, help="Correlation threshold value for creating ad edge in the graph ")
        parser.add_argument( get_opt( ArgparseParameterNames.GA_NRUNS ), type=int, default=10, help="Number of genetic algorithms to be run")
        parser.add_argument( get_opt( ArgparseParameterNames.GA_NGEN ), type=int, default=1000, help="Number of generations for a genetic algorithm run")
        parser.add_argument( get_opt( ArgparseParameterNames.GA_POPSIZE ), type=int, default=100, help="Population size for a genetic algorithm run")
        parser.add_argument( get_opt( ArgparseParameterNames.EV_LOO ), action="store_true")
        parser.add_argument( get_opt( ArgparseParameterNames.GA_MAX_NF ), type=int, default=13, help="Maximum number of features to be selected")
        parser.add_argument( get_opt( ArgparseParameterNames.FLAG_DISABLE_EVAL ), action="store_true", help="Disable feature evaluation after feature selection by GA" )

        parser.add_argument( get_opt( ArgparseParameterNames.GA_PROB_CROSSOVER), type=float, default=0.6, help="Crossover probability" )
        parser.add_argument( get_opt( ArgparseParameterNames.GA_PROB_MUTATION), type=float, default=0.02, help="Crossover probability" )
        
    
    @classmethod
    def set_tuning_parameters(cls, parser: argparse.ArgumentParser):
        # cls.get_standard_parser( parser )
        parser.add_argument( get_opt( ArgparseParameterNames.TU_TRAINED_MODELS ), type=str, required=False)
        parser.add_argument( get_opt( ArgparseParameterNames.TU_EXHAUSTIVE_SEARCH ), action="store_true", help="Exhaustive search for hyperparameters")
    

    @classmethod
    def set_plotgraphs_parameters(cls, parser: argparse.ArgumentParser ):
        parser.add_argument( get_opt( ArgparseParameterNames.PG_SUBGRAPH_ONLY ), action="store_true" )


    @classmethod
    def get_standard_parser(cls, parser: argparse.ArgumentParser, estimators: List[str] = None):
        cls.__set_io_parameters( parser ) 
        cls.__set_cmp_parameters( parser )
        cls.__set_run_parameters( parser, estimators )
        cls.__set_graph_parameters( parser )

    @classmethod
    def __set_graph_parameters(cls, parser: argparse.ArgumentParser):
        parser.add_argument( get_opt( ArgparseParameterNames.GA_EDGE_THRESHOLD ), type=float, default=0.05, help="P-value threshold to build an edge between two features") 
        parser.add_argument( get_opt( ArgparseParameterNames.GRAPH_VW_SCORES ), type=str, nargs="+", default=[ SupportedScores.T_TEST.value ], help=f"List of score functions to be used to set vertices weights; available: {[ x.value for x in SupportedScores ]}")
        parser.add_argument( get_opt( ArgparseParameterNames.GRAPH_EW_SCORE ), type=str, default=SupportedEdgeScorer.PEARSON_R.value, help=f"Score function to be used to create edges: {[ x.value for x in SupportedEdgeScorer ]}")

    @classmethod
    def __set_io_parameters(cls, parser: argparse.ArgumentParser):
        ### Output folders
        parser.add_argument("-o", f"--{FeatSEECore.ParserParameters.output_folder()}", type=str, required=True, help="The output folder (may not exist)")       #output folder (may not exist)
        ### Input files
        parser.add_argument("-i", f"--{FeatSEECore.ParserParameters.input_dataset()}", type=str, nargs="+", help="Input dataset to be used as training (and test) set", required=True)      #input dataset 
        parser.add_argument("-v", f"--{FeatSEECore.ParserParameters.test_sets()}", type=str, nargs="*", default=list(), help="List of dataset to be used as test sets")     #list of test sets 
        parser.add_argument("-f", f"--{FeatSEECore.ParserParameters.feature_sets()}", type=str, nargs="+", default=list(), help="List of features sets to be considered")       #list of feature lists 

    @classmethod
    def __set_cmp_parameters(cls, parser: argparse.ArgumentParser):
        parser.add_argument("-t", get_opt( ArgparseParameterNames.TARGET_FEATURE ), type=str, required=True, help="Name of the categorical feature to be set as target")          #name of the (categorical) feature to be predicted 
        parser.add_argument("-p", get_opt( ArgparseParameterNames.TARGET_POS_LABELS ), type=str, nargs="+", help="Values of the target feature to be considered as positive examples")          #labels to be considered as positive     
        parser.add_argument("-n", get_opt( ArgparseParameterNames.TARGET_NEG_LABELS ), type=str, nargs="+", help="Values of the target feature to be considered as negative examples")          #labels to be considered as negative


    @classmethod
    def __set_run_parameters(cls, parser: argparse.ArgumentParser, estimators: List[str] = None):
        ###################### 
        parser.add_argument( get_opt( ArgparseParameterNames.NUM_PROCESSES ), type=int, default=4, help="Number of processes to be used during cross-validation procedure")
        parser.add_argument( get_opt( ArgparseParameterNames.EV_NTRIALS ), type=int, default=2, help="How many times to repeat the cross-validation procedure")                    #num of runs to be done 
        parser.add_argument( get_opt( ArgparseParameterNames.EV_NCV ), type=int, default=10, help="Number of folds during cross-validation")                      #number of folds to be used during cross validation 
        parser.add_argument( get_opt( ArgparseParameterNames.TSET_PROPORTION ), type=float, required=False, default=0, help="Proportion of the input dataset to be used as test set")   #dataset proportion to be used as test set 
        ######################
        estimator_helper = "List of estimator to be used during feature evaluation."
        default_estimators = [ "ALL" ]

        if estimators is not None: 
            default_estimators = estimators
            estimator_helper += f" Available ones: {', '.join( estimators )}"
        
        parser.add_argument( 
            get_opt( ArgparseParameterNames.ESTIMATORS), 
            type=str, nargs="+", 
            default = default_estimators, 
            help = estimator_helper )          #list of estimators to be used




class InputItem:
    def __init__(self, item: str) -> None:
        self.__item = os.path.abspath( item ) 
        self.__mounted_item = None 

    @property
    def name(self):
        return self.__item
    
    @property
    def mounted_name(self):
        return self.__mounted_item

    
    def exists(self):
        return os.path.exists( self.__item )
    
    def mount(self, mountpoint: str):
        self.__mounted_item = os.path.join(
            mountpoint, os.path.basename( self.__item )
        )

    def copy_in(self, dest_folder: str):

        os.makedirs( dest_folder, exist_ok=True )
        if os.path.isfile( self.__item ):
            basename_f = os.path.basename( self.__item )
            #copy file and attach it to the new name to the proper list 
            shutil.copyfile(self.__item, os.path.join(dest_folder, basename_f)) #copy file in the docker folder
        
        elif os.path.isdir( self.__item ):
            new_dir_name = os.path.split( self.__item.rstrip("/") )[-1]
            new_dir = os.path.join(dest_folder, new_dir_name)

            shutil.copytree(self.__item, os.path.join(dest_folder, new_dir))


class InputGroup:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files = list() 
    
    def add(self, new_elems: Iterable[str]):
        if isinstance(new_elems, str):
            self.files.append( InputItem( new_elems ) )
        else:
            self.files.extend([ InputItem( f ) for f in new_elems ])
            
    def check_input(self) -> List:
        """ Check files for existence: return a list containing 
        filenames which has not be found: """

        wrong_files = list()
        check = { f.name: f.exists() for f in self.files }

        if not all( check.values() ):
            wrong_files = [f for f, f_exists  in check.items() if not f_exists]
            
        return wrong_files
    
    def mount_files(self, docker_mountpoint: str, dest_folder: str):
        for f in self.files:
            f.mount( docker_mountpoint )
            f.copy_in( dest_folder )



class FeatSEECore:
    class ParserParameters:
        @classmethod
        def input_dataset(cls):
            return ArgparseParameterNames.INPUT_DATASET.value
        
        @classmethod
        def output_folder(cls):
            return ArgparseParameterNames.OUTPUT_FOLDER.value

        @classmethod
        def test_sets(cls):
            return ArgparseParameterNames.TEST_SETS.value
        
        @classmethod
        def feature_sets(cls):
            return ArgparseParameterNames.FEATURE_SETS.value

        @classmethod
        def rules_sets(cls):
            return ArgparseParameterNames.RULES_SETS.value

        @classmethod
        def trained_models(cls):
            return ArgparseParameterNames.TRAINED_MODELS.value
            # return "trained_models"


    class CommandHelper( enum.Enum ):
        EVALUATION = "Evaluate 1+ sets of features"
        EXPLANATION = "Explain the features of a trained model"
        SELECTION = "Select features from a training set"
        GA_SELECTION = "Run a genetic algorithm to perform feature selection"
        TUNING = "Tune the parameters of 1+ model(s)"
        EDA = "Perform exploratory data analysis on the provided dataset"
        PLOT_GRAPHS = "Show feature correlation graphs"


    class SupportedAction(enum.Enum):
        EVALUATION = "evaluate"
        SELECTION = "select"
        EXPLAINATION = "explain"
        GA_SELECTION = "GA" 
        TUNING = "tune"     
        EDA = "EDA"
        PLOT_GRAPHS = "plots"

        @classmethod
        def from_command(cls, command: str):
            try:
                return list( filter( lambda x: x.value == command, cls ) )[0]
            except IndexError:
                raise ValueError(f"Command {command} not supported")


    DOCKER_IMAGE = "cursecatcher/featsee"

    INPUT_ATTRS = (
        "command",

        ParserParameters.output_folder(), 

        ParserParameters.input_dataset(), 
        ParserParameters.test_sets(), 
        ParserParameters.feature_sets(), 
        ParserParameters.rules_sets(), 
        ParserParameters.trained_models()
    )


    def __init__(self, parser: argparse.ArgumentParser) -> None:        
        self.__args = parser.parse_args(sys.argv[1:])
        self.__to_do = self.SupportedAction.from_command( self.__args.command )

        self.__training_data = InputGroup( self.ParserParameters.input_dataset() )   #training data
        self.__test_sets = InputGroup( self.ParserParameters.test_sets() )           #test sets
        self.__features = InputGroup( self.ParserParameters.feature_sets() )        #features sets
        self.__rules = InputGroup( self.ParserParameters.rules_sets() )              #rules sets
        self.__models = InputGroup( self.ParserParameters.trained_models() )         #fitted models

        self.__docker_outfolder = os.path.abspath( self.__args.outfolder )
        self.__docker_mountdata = "/data"

        self.__image_tag = "latest"
        self.__container_name = None 


        self.__all_data = [
            self.__features, 
            self.__rules,
            self.__training_data, 
            self.__test_sets, 
            self.__models
        ]


        self.__get_input()
        self.__check_input()


    @property
    def image_tag(self):
        return self.__image_tag

    @property
    def container_name(self):
        return self.__container_name

    @image_tag.setter
    def image_tag(self, new_tag: str):
        if new_tag is not None:
            self.__image_tag = new_tag

    @container_name.setter
    def container_name(self, c_name: str):
        if c_name is not None:
            self.__container_name = c_name


    def get_args(self):
        return self.__args

    def __get_input(self):
        self.__training_data.add( self.__args.input_data )

        self.__test_sets.add( self.__args.test_sets )
        self.__features.add( self.__args.feature_lists )

        if self.__to_do is self.SupportedAction.EXPLAINATION:
            self.__rules.add( self.__args.rules )
        
        elif self.__to_do is self.SupportedAction.TUNING and self.__args.load_from:
            self.__models.add( self.__args.load_from )


    def __check_input(self):
        for filelist in self.__all_data:
            check_value = filelist.check_input() 
            
            if check_value:
                print(f"Files not found in group '{filelist.name}': {check_value}")


    def mount_files(self):
        for filelist in self.__all_data:
            filelist.mount_files( 
                os.path.join( self.__docker_mountdata, "in", filelist.name), 
                os.path.join( self.__docker_outfolder, "in", filelist.name ) )


    def __build_docker_command(self):
        def unroll_arg( arg ):
            if isinstance(arg, Iterable) and not isinstance(arg, str):
                return " ".join( arg )
            else:
                return arg 

        def unroll_filelist( argname: str, files: Iterable[InputItem] ):
            if files:
                concat = " ".join( [f.mounted_name for f in files] )
                return f"--{argname} {concat}" 
            return ""

        ## run docker as current user
        get_id = f"$(id -u {getpass.getuser()})"
        self.cidfile = os.path.join( self.__docker_outfolder, "dockerID" )

        ## get arguments providing files (datasets, features, etc)
        argumentz = [ unroll_filelist( x.name, x.files ) for x in self.__all_data ]

        ## get arguments providing other parameters 
        for x, y in vars(self.__args).items():
            if y and x not in self.INPUT_ATTRS:
                if isinstance(y, bool) and y:
                    argumentz.append( f"--{x}" )  ## flag (e.g. store_true / store_false )
                else:
                    argumentz.append( f"--{x} {unroll_arg( y )}" )

        argumentz = " ".join( argumentz )

        set_container_name = f"--name {self.__container_name}" if self.__container_name else ""


        ### COMMAND TEMPLATE: 
        # docker run [OPTIONS] -v <volumes...> IMAGE:TAG COMMAND ARGS... 
        command = f"""
            docker run -d {set_container_name} --shm-size=5gb --cidfile {self.cidfile}  -u {get_id}:{get_id}
               -v {self.__docker_outfolder}:/data/out -v {self.__docker_outfolder}/in:/data/in
                {self.DOCKER_IMAGE}:{self.__image_tag} featsee.py {self.__args.command} 
                    -o /data/out {argumentz}"""
        
        return command.replace("\n", " ").replace("\t", " ").strip()



            
    def run_container(self, verboseness: bool = False):
        command = self.__build_docker_command()

        print(f"Running the docker container w/ the following command:\n\n{command}\n")

        sp = subprocess.run(command, shell=True)
        
        while not os.path.exists(self.cidfile):
            time.sleep(1)
        
        with open(self.cidfile) as f:
            cid = f.readline().strip()

        if sp.returncode != 0:
            sys.exit(f"Eh errore ")


        if verboseness:
            print("Showing current execution. Press ctrl+c to quit.")
            try:
                subprocess.run(f"docker logs -f {cid}", shell=True)
            except KeyboardInterrupt:
                print(f"\nOk, bye.\nPs. your container ID is {cid}")





class ParserFeatSEE:
    
    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser = None, avail_estimators: List[str] = None) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser("FeatSEE: Feature Selection, Evaluation and Explaination")

        subparsers_details = [
            ( Parser.set_evaluation_parameters, FeatSEECore.SupportedAction.EVALUATION, FeatSEECore.CommandHelper.EVALUATION ), 
            ( Parser.set_explaination_parameters, FeatSEECore.SupportedAction.EXPLAINATION, FeatSEECore.CommandHelper.EXPLANATION ), 
            ( Parser.set_fselection_parameters, FeatSEECore.SupportedAction.SELECTION, FeatSEECore.CommandHelper.SELECTION ), 
            ( Parser.set_ga_parameters, FeatSEECore.SupportedAction.GA_SELECTION, FeatSEECore.CommandHelper.GA_SELECTION ), 
            ( Parser.set_tuning_parameters, FeatSEECore.SupportedAction.TUNING, FeatSEECore.CommandHelper.TUNING ), 
            ( None, FeatSEECore.SupportedAction.EDA, FeatSEECore.CommandHelper.EDA ),
            ( Parser.set_plotgraphs_parameters, FeatSEECore.SupportedAction.PLOT_GRAPHS, FeatSEECore.CommandHelper.PLOT_GRAPHS ),
        ]
        subparser = parser.add_subparsers(dest="command", title="Commands", description="Valid commands", required=True)

        for parser_method, operation, helper in subparsers_details:
            #create a new subparser using the current operation and helper message 
            curr_parser = subparser.add_parser( operation.value, help = helper.value )
            # set standard parameters, then customize the parser wrt current operation 
            Parser.get_standard_parser( curr_parser, avail_estimators )

            if parser_method is not None: 
                parser_method( curr_parser )

        return parser 





class Batcher:
    """ Class to create a batch of jobs. """

    class Filenames( enum.Enum ):
        """ Just a class to store the names of the files created by the batcher. """

        BATCH = "batch.tsv"
        PARAMS = "params.json"
        BASH_SCRIPT = "bashino.sh" # TODO:  maybe run the script ? 


    class BatchfileManager:
        """ Class to manage the batch file. """

        class Parameters2Run:
            """ Represent the parameters for a classification task: 
            - the target feature T
            - the 1+ values of T representing the positive class
            - the 1+ values of T representing the negative class """


            HEADER = [ 
                "target_feature", "pos_class", "neg_class"
            ]

            def __init__(self, params: tuple) -> None:
                assert len(params) == 3 #(exactly) 3 parameters: (target_feature, pos_class, neg_class)
                self.feature = params[0]
                self.pos_class = self.__unroll( params[1] )
                self.neg_class = self.__unroll( params[2] )
                
            def __unroll(self, param: str) -> list:
                """ Unroll a parameter that may be a comma-separated list of values. """
                return [ x.strip() for x in param.split(",") ]


        def __init__(self, batch_folder: str = None) -> None:
            self.__batchfolder = batch_folder
            self.__batchlist = list() 
        
        @property
        def batchfolder(self):
            return self.__batchfolder
        
        @batchfolder.setter
        def batchfolder(self, batchfolder: str):
            self.__batchfolder = batchfolder
        
        def __get_batch_filename(self):
            return os.path.join( self.__batchfolder, Batcher.Filenames.BATCH.value )
        
        def read(self):
            with open( self.__get_batch_filename(), "r" ) as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader) #skip header

                self.__batchlist = [ self.Parameters2Run(row) for row in reader if row ]

        def write(self):
            with open( self.__get_batch_filename(), "w" ) as fbatch:
                writer = csv.writer( fbatch, delimiter="\t" )
                writer.writerow( self.Parameters2Run.HEADER )


        def __iter__(self):
            return iter(self.__batchlist)




    class ParameterSet:
        """ Class to manage the parameters of the batch. """

        def __init__(self, op: str = None) -> None:
            self.__op = op 
            self.__generic = dict() 
            self.__specific = dict()

        @property
        def operation(self):
            return self.__op

        @property
        def generics(self):
            return self.__generic
        
        @property
        def specific(self):
            return self.__specific

        
        def __format_general_params(self):
            """ Rebuild generic parameters dictionary by setting the proper parameters' names """

            keys = [ 
                ArgparseParameterNames.INPUT_DATASET, 
                ArgparseParameterNames.FEATURE_SETS, 
                ArgparseParameterNames.TEST_SETS, 
                ArgparseParameterNames.EV_NTRIALS, 
                ArgparseParameterNames.EV_NCV, 
                ArgparseParameterNames.TSET_PROPORTION, 
                ArgparseParameterNames.ESTIMATORS 
            ]

            return ArgparseParameterNames.replace_keys({
                k: self.generics.get( k.value ) for k in keys
            })
            

        def format_params(self, cmp_params: dict):
            def unroll_list( arg ):
                if isinstance( arg, list ):
                    return " ".join( arg )
                elif isinstance( arg, bool ):
                    return ""
                else:
                    return arg 
                    

            def unpack_dict( d: dict ):
                l = [ f"--{k} {unroll_list( v )}" for k, v in d.items() if v ]
                return " ".join(l)

            all_args = [ self.__format_general_params(), self.__specific, cmp_params ]
            general, specific, cmp = [ unpack_dict( curr_args ) for curr_args in all_args ]

            return f"{self.__op} {general} {cmp} {specific}"

        #### XXX replace args names with enums
        def set_specific(self):
            if self.__op == FeatSEECore.SupportedAction.EVALUATION.value:
                self.__specific = {
                    ArgparseParameterNames.EV_LOO: False
                }

            elif self.__op == FeatSEECore.SupportedAction.SELECTION.value:
                self.__specific = {
                    ArgparseParameterNames.FS_MIN_NF: 1, 
                    ArgparseParameterNames.FS_MAX_NF: 10, 
                    ArgparseParameterNames.FS_MIN_AUC: 0.7, 
                    ArgparseParameterNames.FS_NTEST_EVAL: 3, 
                    ArgparseParameterNames.FS_NCV_EVAL: 10 
                }
            
            elif self.__op == FeatSEECore.SupportedAction.GA_SELECTION.value:
                self.__specific = {
                    ArgparseParameterNames.GA_EDGE_THRESHOLD: 0.5, 
                    ArgparseParameterNames.GA_NGEN: 50, 
                    ArgparseParameterNames.GA_NRUNS: 10, 
                    ArgparseParameterNames.GA_POPSIZE: 30, 
                    ArgparseParameterNames.GA_MAX_NF: None
                }

            else:
                raise NotImplementedError( self.__op ) #Exception("BOH ", self.__op)

            self.__specific = ArgparseParameterNames.replace_keys( self.__specific )

        def set_generic(self, datasets: list, features: list, test_sets: list ):
            def get_abs_path( filenames: list ):
                return [ os.path.abspath( x ) for x in filenames ]


            self.__generic = ArgparseParameterNames.replace_keys({
                ArgparseParameterNames.INPUT_DATASET: get_abs_path( datasets ),
                ArgparseParameterNames.FEATURE_SETS: get_abs_path( features ), 
                ArgparseParameterNames.TEST_SETS: get_abs_path( test_sets ), 
                ArgparseParameterNames.EV_NTRIALS: 10, 
                ArgparseParameterNames.EV_NCV: 10, 
                ArgparseParameterNames.TSET_PROPORTION: 0.3, 
                ArgparseParameterNames.ESTIMATORS: "ALL"
            })


        def to_json(self, filename):
            """ Write the parameters to a JSON file. """
            content = dict( 
                generics = self.__generic, 
                specifics = self.__specific, 
                operation = self.__op
            )

            with open( filename, "w" ) as f:
                json.dump( content, f, indent=4 )
            
            return self 
        

        def load_json(self, filename):
            """ Load the parameters from a JSON file. """
            with open( filename, "r" ) as f:
                tmp = json.load( f )

                self.__generic = tmp.get( "generics" )
                self.__specific = tmp.get( "specifics" )
                self.__op = tmp.get( "operation" )
            
            return self 


    def __init__(self, parsed_args: argparse.Namespace ) -> None:
        self.__args = parsed_args
        self.__bmanager = self.BatchfileManager()
        self.__params = None 


    def init_project(self):

        outfolder, in_datasets, f_sets, t_sets = ArgparseParameterNames.build_list_args(
            vars( self.__args ), 
            map( lambda k: k.value, [ 
                ArgparseParameterNames.OUTPUT_FOLDER, ArgparseParameterNames.INPUT_DATASET, 
                ArgparseParameterNames.FEATURE_SETS, ArgparseParameterNames.TEST_SETS ] 
            )
        )
        op = self.__args.op 

        ## build project folder
        os.makedirs( outfolder, exist_ok=False )
        self.__bmanager.batchfolder = outfolder
        self.__bmanager.write()

        self.__params = self.ParameterSet( op )
        self.__params.set_generic( in_datasets, f_sets, t_sets )
        self.__params.set_specific()
        self.__params.to_json( os.path.join( outfolder, self.Filenames.PARAMS.value ) )
        

    def run_batch(self):
        """ Create a bash file with all the commands to run. """

        outfolder = self.__args.batch 
        batch_name = os.path.basename( os.path.abspath( outfolder ) ) 
        biodocker_path = os.path.abspath( __file__ )

        self.__bmanager.batchfolder = outfolder  # set batch folder 
        self.__bmanager.read() # get info from batch file

        self.__params = self.ParameterSet().load_json( 
            os.path.join( outfolder, self.Filenames.PARAMS.value ) )

        ## iterate over run parameters of different configurations (target feature, classes to predict)
        biodocker_to_run = list() 

        for cmp in self.__bmanager:

            pos_labels, neg_labels = [ 
                "_".join( classes ) for classes 
                    in (cmp.pos_class, cmp.neg_class)  ]

            session_folder = f"{cmp.feature}__{pos_labels}_vs_{neg_labels}"
            container_name = f"{self.__params.operation}_{batch_name}_{session_folder}"

            curr = dict(
                target = cmp.feature,
                pos_labels = cmp.pos_class,
                neg_labels = cmp.neg_class,
                outfolder = session_folder #os.path.join( outfolder, session_folder )
            )

            cmp_name = f"# TARGET of CLASSIFICATION => {cmp.feature}\t- {pos_labels} vs {neg_labels}\n\n"
            biodocker_call = f"{biodocker_path} docker --container_name {container_name} {self.__params.format_params( curr )}"
            biodocker_to_run.append( ( cmp_name, biodocker_call ) )

        bashfilename = os.path.join( outfolder, self.Filenames.BASH_SCRIPT.value )

        with open( bashfilename, "w" ) as fbash:
            fbash.write("#!/bin/bash\n\n")
            # fbash.writelines( bash_lines )

            for cmp_name, biodocker_call in biodocker_to_run:
                fbash.write( cmp_name )
                fbash.write( biodocker_call )
                fbash.write("\n\n")

        fstats = os.stat( bashfilename )
        os.chmod( bashfilename, fstats.st_mode | stat.S_IEXEC )

        for name, b2r in biodocker_to_run:
            print(f"\nRunning {name}: ")
            subprocess.run( b2r, shell = True, check = True, cwd = outfolder )


    @classmethod
    def get_parser(self, parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
        if not parser:
            parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", title="FeatSEE Batcher")

        avail_operations = [v.value for v in FeatSEECore.SupportedAction]

        #### Initialize a new project ####
        parser_init = subparsers.add_parser("init",
            add_help=False, description="Evaluate feature sets using different ML models",  help="...")
        #### XXX REPLACE STRINGS WITH ENUMS 
        # "--input"
        parser_init.add_argument("-i",  get_opt( ArgparseParameterNames.INPUT_DATASET ), nargs="+", type=str,  required=True, help="Input files")
        # "--output"
        parser_init.add_argument("-o", get_opt( ArgparseParameterNames.OUTPUT_FOLDER ), type=str, required=True, help="Output directory")
        # "--test_sets", 
        parser_init.add_argument("-t", get_opt( ArgparseParameterNames.TEST_SETS ), nargs="*", type=str, default=list(), help="Test sets")
        # "--features"
        parser_init.add_argument("-f", get_opt( ArgparseParameterNames.FEATURE_SETS ), nargs="*", type=str, default=list(), help="Feature sets")
        parser_init.add_argument("--op", choices=avail_operations, default=FeatSEECore.SupportedAction.EVALUATION.value, help="Operation to perform")
        
        #### Run analyses in a project ####
        parser_run = subparsers.add_parser("run",
            add_help=False, description="Run batch of analyses", help="...")
        parser_run.add_argument("-b", "--batch", type=str, required=True, help="Batch folder")


        return parser



class MainCommand( enum.Enum ):
    BATCH = "batch"
    DOCKER = "docker"




if __name__ == "__main__":
    parser = argparse.ArgumentParser("FeatSEE")

    subparsers = parser.add_subparsers(help='Run a single analysis using docker or a batch on them?', dest="mode")
    # create the parser to run analyses using docker
    parser_docker = subparsers.add_parser(
        MainCommand.DOCKER.value, 
        help='Run a dockerized analysis')

    ## docker options 
    parser_docker.add_argument("--verbose", action="store_true", help="Show docker logs while running")
    parser_docker.add_argument("--tag", type=str, help="Docker image tag")
    parser_docker.add_argument("--container_name", type=str, required=False, help="Docker container name")
    ## FeatSEE options
    parser_docker = ParserFeatSEE.get_parser( parser_docker )

    # create the parser to prepare a batch of analyses
    parser_batcher = subparsers.add_parser(
        MainCommand.BATCH.value, 
        help='Run a batch of dockerized analyses')
        
    Batcher.get_parser( parser_batcher )
    
    args = parser.parse_args()
    
    
    if args.mode == MainCommand.DOCKER.value:
        featsee = FeatSEECore( parser )

        args = featsee.get_args() 

        featsee.image_tag = args.tag
        featsee.container_name = args.container_name

        featsee.mount_files()
        featsee.run_container( args.verbose )
    
    elif args.mode == MainCommand.BATCH.value:
        batcher = Batcher(args)

        if args.command == "init":
            batcher.init_project()

        elif args.command == "run":
            batcher.run_batch()

        else:
            parser.print_help()

