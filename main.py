import yaml
from pipeline.pipeline import TitanicKernelSVMPipeline


class TitanicKernelSVMMain:

    def __init__(self):
        self.train_ds_path = None
        self.test_ds_path = None
        self.output_path = None

    def yaml_loader(self):
        with open('conf/model-properties.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.train_ds_path = str(config['environment']['model-arguments']['train_ds_path'])
        self.test_ds_path = str(config['environment']['model-arguments']['test_ds_path'])
        self.output_path = str(config['environment']['model-arguments']['output_path'])
    
    def start(self):
        self.yaml_loader()
        TitanicKernelSVMPipeline(
            train_ds_path=self.train_ds_path,
            test_ds_path=self.test_ds_path,
            output_path=self.output_path
        ).process()


if __name__ == '__main__':
    TitanicKernelSVMMain().start()
