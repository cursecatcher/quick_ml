#!/usr/bin/env python3 

import argparse
import csv, json
import stat, sys
from typing import Iterable, List 
import os, shutil, getpass, enum, subprocess, time 


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
            return "input_data"
        
        @classmethod
        def output_folder(cls):
            return "outfolder"

        @classmethod
        def test_sets(cls):
            return "test_sets"
        
        @classmethod
        def feature_sets(cls):
            return "feature_lists"

        @classmethod
        def rules_sets(cls):
            return "rules"

        @classmethod
        def trained_models(cls):
            return "load_from" ## load_models ? 
            # return "trained_models"


    class SupportedAction(enum.Enum):
        EVALUATION = "evaluate"
        SELECTION = "select"
        EXPLAINATION = "explain"
        GA_SELECTION = "GA" 
        TUNING = "tune"     

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
            docker run -d {set_container_name} --cidfile {self.cidfile}  -u {get_id}:{get_id}
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



class Parser:
    """ Auxiliary methods to parse command line arguments """

    @classmethod
    def set_evaluation_parameters(cls, parser: argparse.ArgumentParser):
        cls.get_standard_parser( parser )

        parser.add_argument("--loo", action="store_true", help="Enable leave-one-out cross-validation")

    @classmethod
    def set_fselection_parameters(cls, parser: argparse.ArgumentParser):
        cls.get_standard_parser( parser ) 

        parser.add_argument("--min_nf", type=int, default=1, help="Minimum number of features to be selected")
        parser.add_argument("--max_nf", type=int, help="Maximum number of features to be selected")
        parser.add_argument("--min_auc", type=float, required=False, default=0.8, help="Minimum AUC to be considered")
        # parser.add_argument("--score", type=str, default="pearson")

        parser.add_argument("--ntrials_test", type=int, default=3, help="Number of trials during feature evaluation phase")
        parser.add_argument("--ncv_test", type=int, default=10, help="Number of cross-validation folds during feature evaluation phase")


    @classmethod
    def set_explaination_parameters(cls, parser: argparse.ArgumentParser):
        cls.get_standard_parser( parser )

        parser.add_argument("-r", "--rules", type=str, nargs="*")
        parser.add_argument("-c", "--clusters", type=int, default=3)
        parser.add_argument("--ntrials_test", type=int, default=3)
        parser.add_argument("--ncv_test", type=int, default=10)
        parser.add_argument("--loo", action="store_true")
        parser.add_argument("--max_rules", type=int, required=False)

    @classmethod
    def set_ga_parameters(cls, parser: argparse.ArgumentParser):
        cls.get_standard_parser( parser )

        parser.add_argument("--edge_threshold", type=float, default=0.5, help="Correlation threshold value for creating ad edge in the graph ")
        parser.add_argument("--nga", type=int, default=10, help="Number of genetic algorithms to be run")
        parser.add_argument("--ngen", type=int, default=1000, help="Number of generations for a genetic algorithm run")
        parser.add_argument("--popsize", type=int, default=100, help="Population size for a genetic algorithm run")
        parser.add_argument("--loo", action="store_true")
        # parser.add_argument("--score", type=str, default="pearson", help="Score to be used for feature selection (e.g. pearson, anova)")
        parser.add_argument("--max_nf", type=int, help="Maximum number of features to be selected")


    
    @classmethod
    def set_tuning_parameters(cls, parser: argparse.ArgumentParser):
        cls.get_standard_parser( parser )
        parser.add_argument("--load_from", type=str, required=False)
        parser.add_argument("--exhaustive", action="store_true", help="Exhaustive search for hyperparameters")
    


    @classmethod
    def get_standard_parser(cls, parser: argparse.ArgumentParser):
        cls.__set_io_parameters( parser ) 
        cls.__set_cmp_parameters( parser )
        cls.__set_run_parameters( parser )

    @classmethod
    def __set_io_parameters(cls, parser: argparse.ArgumentParser):
        ### Output folders
        parser.add_argument("-o", f"--{FeatSEECore.ParserParameters.output_folder()}", type=str, required=True, help="The output folder (may not exist)")       #output folder (may not exist)
        ### Input files
        parser.add_argument("-i", f"--{FeatSEECore.ParserParameters.input_dataset()}", type=str, nargs="+", help="Input dataset to be used as training (and test) set")      #input dataset 
        parser.add_argument("-v", f"--{FeatSEECore.ParserParameters.test_sets()}", type=str, nargs="*", default=list(), help="List of dataset to be used as test sets")     #list of test sets 
        parser.add_argument("-f", f"--{FeatSEECore.ParserParameters.feature_sets()}", type=str, nargs="+", default=list(), help="List of features sets to be considered")       #list of feature lists 

    @classmethod
    def __set_cmp_parameters(cls, parser: argparse.ArgumentParser):
        parser.add_argument("-t", "--target", type=str, required=True, help="Name of the categorical feature to be set as target")          #name of the (categorical) feature to be predicted 
        parser.add_argument("-p", "--pos_labels", type=str, nargs="+", help="Values of the target feature to be considered as positive examples")          #labels to be considered as positive     
        parser.add_argument("-n", "--neg_labels", type=str, nargs="+", help="Values of the target feature to be considered as negative examples")          #labels to be considered as negative


    @classmethod
    def __set_run_parameters(cls, parser: argparse.ArgumentParser):
        ###################### 
        parser.add_argument("--trials", type=int, default=2, help="How many times to repeat the cross-validation procedure")                    #num of runs to be done 
        parser.add_argument("--ncv", type=int, default=10, help="Number of folds during cross-validation")                      #number of folds to be used during cross validation 
        parser.add_argument("--tsize", type=float, required=False, default=0, help="Proportion of the input dataset to be used as test set")   #dataset proportion to be used as test set 
        ######################
        parser.add_argument("--estimators", type=str, nargs="+", default=["ALL"], help="List of estimator to be used during feature evaluation")          #list of estimators to be used



class ParserFeatSEE:
    
    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser("FeatSEE: Feature Selection, Evaluation and Explaination")

        subparser = parser.add_subparsers(dest="command", title="Commands", description="Valid commands", required=True)

        Parser.set_evaluation_parameters( subparser.add_parser( 
                FeatSEECore.SupportedAction.EVALUATION.value, 
                help="Evaluate 1+ sets of features") )

        Parser.set_explaination_parameters( subparser.add_parser(
                FeatSEECore.SupportedAction.EXPLAINATION.value, 
                help="Explain the features of a trained model") )

        Parser.set_fselection_parameters( subparser.add_parser(
                FeatSEECore.SupportedAction.SELECTION.value,
                help="Select features from a training set") )

        Parser.set_ga_parameters( subparser.add_parser(
                FeatSEECore.SupportedAction.GA_SELECTION.value,
                help="Run a genetic algorithm to perform feature selection") )

        Parser.set_tuning_parameters( subparser.add_parser(
                FeatSEECore.SupportedAction.TUNING.value,
                help="Tune the parameters of 1+ model(s)") )

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
            return dict(
                #Input data 
                input_data = self.generics.get( "input_datasets" ), 
                feature_lists = self.generics.get( "feature_sets" ),
                test_sets = self.generics.get( "test_sets" ),
                # Training parameters 
                trials = self.generics.get( "n_trials" ),
                ncv = self.generics.get( "n_folds" ),
                tsize = self.generics.get( "test_size" ), 
                estimators = self.generics.get( "estimators" )
            )

        def format_params(self, cmp_params: dict):
            def unroll_list( arg ):
                return " ".join( arg ) if isinstance( arg, list ) else arg 

            def unpack_dict( d: dict ):
                l = [ f"--{k} {unroll_list( v )}" for k, v in d.items() if v ]
                return " ".join(l)

            general = unpack_dict( self.__format_general_params() )
            specific = unpack_dict( self.__specific )
            cmp = unpack_dict( cmp_params )

            return f"{self.__op} {general} {cmp} {specific}"


        def set_specific(self):
            if self.__op == FeatSEECore.SupportedAction.EVALUATION.value:
                self.__specific = dict( loo = False )

            elif self.__op == FeatSEECore.SupportedAction.SELECTION.value:
                self.__specific = dict( 
                    min_nf = 1,
                    max_nf = 10, 
                    min_auc = 0.7, 
                    score = "pearson", 
                    ntrials_test = 3, 
                    ncv_test = 10 )
            
            elif self.__op == FeatSEECore.SupportedAction.GA_SELECTION.value:
                self.__specific = dict(
                    edge_threshold = 0.5,
                    ngen = 50,
                    nga = 10,
                    popsize = 30, 
                    max_nf = None
                    # score = "pearson"
                )


            else:
                raise NotImplementedError( self.__op ) #Exception("BOH ", self.__op)


        def set_generic(self, datasets: list, features: list, test_sets: list ):
            def get_abs_path( filenames: list ):
                return [ os.path.abspath( x ) for x in filenames ]

            self.__generic = dict(
                input_datasets =  get_abs_path( datasets ), 
                feature_sets = get_abs_path( features ), 
                test_sets = get_abs_path( test_sets ), 
                n_trials = 10, 
                n_folds = 10, 
                test_size = 0.3, 
                estimators = "ALL"
            )

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
        outfolder = self.__args.output
        in_datasets = self.__args.input 
        f_sets = self.__args.features 
        t_sets = self.__args.test_sets 
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
        parser_init.add_argument("-i", "--input", nargs="+", type=str,  required=True, help="Input files")
        parser_init.add_argument("-o", "--output", type=str, required=True, help="Output directory")
        parser_init.add_argument("-t", "--test_sets", nargs="*", type=str, default=list(), help="Test sets")
        parser_init.add_argument("-f", "--features", nargs="*", type=str, default=list(), help="Feature sets")
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

