import fasttext
import matplotlib.pyplot as plt
import numpy as np
from inspect import stack
from random import randint
from typing import Dict
import pandas as pd


class OverfitPredictor(object):

    def __init__(self,
                 train_txt_path,
                 validation_txt_path,
                 save_figs_to,
                 steps_for_epochs_=1,
                 use_pretrained_=False):

        self.train_txt_path = train_txt_path
        self.validation_txt_path = validation_txt_path
        self.save_figs_to = save_figs_to
        self.steps_for_epochs = steps_for_epochs_
        self.use_pretrained = use_pretrained_
        self.global_maximum = 0
        self.patience_left = 0

    def start(self,
              _params: dict,
              randomseed=False,
              multipleruns=0
              ) -> dict:
        """
        Function to run the Fasttext model with the parameters given when the Class is instanciated.
        It draws a graph with train/validation [precision, recall, f1] during the expected number of
        epochs, and steps between each training round (given in the classes' constructor).

        :multipleruns: If set, the function calculates the given number of epochs' average
        :return: None
        """

        results = self.__test_overfitting(_params,
                                          randomseed,
                                          multipleruns)

        return results

    def start_with_earlystopping(self,
                                 _params: dict,
                                 monitor: str,
                                 patience: int = 1,
                                 min_delta: float = 0.0,
                                 randomseed=False) -> dict:

        """
        Trains Fasttext model with early stop. If within 'patience' epochs, there is no at least 'min_delta' improvement
        in 'monitor' metric, training stops, and results so far will be plotted.

        :param randomseed: If set True, a random generation seed is initialized in each run of the training for the model
        :param monitor: The metric selected for monitoring, either 'precision', 'recall' or 'f1'
        :param patience: The number of epochs within which, if there is no improvement compared to the global maximum, learning stops.
        :param min_delta: The expected improvement in the 'monitor'-ed metric, in percent (e.g.: for 5% type: '0.05')
        :return: None
        """

        if monitor not in ["precision", "recall", "f1"]:
            raise TypeError((
                "ERROR! Please specify one of the followings as 'monitor' parameter: 'precision', 'recall' or "
                "'f1'"))

        RESULTS = {}

        max_epochs = _params["epoch"]
        self.patience_left = patience
        self.global_maximum = 0

        for epoch in range(self.steps_for_epochs, max_epochs + 1, self.steps_for_epochs):

            _params['epoch'] = epoch
            print(_params["epoch"])
            if randomseed:
                _params["seed"] = randint(1, 100)

            results = self.train_model(_params, 1)
            print(results)

            # elso epoch-ban letaroljuk a metrikat mint globalis maximumot
            if epoch == self.steps_for_epochs:
                if monitor == "precision":
                    self.global_maximum = results['precision_validation'][0]  # precision_validation
                elif monitor == "recall":
                    self.global_maximum = results['recall_validation'][0]  # recall_validation
                else:  # f1
                    current_f1 = self.__calc_f1(results[epoch]['precision_validation'][0],
                                                results[1]['recall_validation'][0])
                    self.global_maximum = current_f1

                RESULTS[epoch] = results

                continue
            # utana megnezzuk, hogy a kivalasztott metrika következő értéke >= -e a globális maximumnál,
            # ha igen, patience reset, új globális maximum set
            # ha nem, pateince -= 1
            # global maximum csak akkor frissul, ha az uj ertek legalabb min_deltá-val nagyobb!!! addig változatlan

            else:
                if monitor == "precision":
                    current = results['precision_validation'][0]
                elif monitor == "recall":
                    current = results['recall_validation'][0]
                else:
                    current = self.__calc_f1(results[epoch]['precision_validation'][0],
                                             results[1]['recall_validation'][0])

                if current >= self.global_maximum + min_delta:
                    # print("GM: " + str(self.global_maximum))
                    self.patience_left = patience  # ha van improvement, patience reset!
                    self.global_maximum = current
                else:
                    if self.patience_left > 0:
                        self.patience_left -= 1
                    else:
                        max_epochs = epoch
                        RESULTS[epoch] = results
                        break

                RESULTS[epoch] = results

        self.plot_results(RESULTS,
                          self.save_figs_to,
                          max_epochs,
                          self.use_pretrained,
                          multipleruns=0)

        return RESULTS

    def __test_overfitting(self, _params, randomseed, multipleruns) -> dict:

        max_epochs = _params["epoch"]

        """
        Internal storage of metrics:
        RESULTS = { 
            epoch_number_1: 
                {
                    "precision_train" :     [results],
                    "recall_train":         [results],
                    "precision_validation": [results],
                    "recall_validation":    [results],
                },
            epoch_number_2: ...
        }
        """

        RESULTS = {}

        for epoch in range(self.steps_for_epochs, max_epochs + 1, self.steps_for_epochs):
            _params['epoch'] = epoch
            print(_params["epoch"])
            if randomseed:
                _params["seed"] = randint(1, 100)

            if multipleruns > 0:
                results = self.train_model(_params, multipleruns)
            else:
                results = self.train_model(_params, 1)

            RESULTS[epoch] = results

        print(RESULTS)

        self.plot_results(RESULTS,
                          self.save_figs_to,
                          max_epochs,
                          self.use_pretrained,
                          multipleruns
                          )

        return RESULTS

    def train_model(self, _params, x_times_average: int) -> Dict:
        """

        :param x_times_average: trains "x_times_average" model with tha same epoch number
        :return: calculates the average in every epoch of trained models, then store the min, max, average in a dict,
        e.g.: {precision_train: [60, 70, 65], recall_train: [] ...}
        """

        result = {}
        precisions_train = []
        recalls_train = []
        precisions_validation = []
        recalls_validation = []

        if x_times_average == 1:
            model = fasttext.train_supervised(**_params)

            samples, precision_train, recall_train = model.test(self.train_txt_path, k=-1, threshold=0.5)
            samples, precision_validation, recall_validation = model.test(self.validation_txt_path, k=-1, threshold=0.5)

            result["precision_train"] = [precision_train, precision_train, precision_train]
            result["recall_train"] = [recall_train, recall_train, recall_train]
            result["precision_validation"] = [precision_validation, precision_validation, precision_validation]
            result["recall_validation"] = [recall_validation, recall_validation, recall_validation]

            classes = model.test_label(validation_txt_path_, k=-1, threshold=0.5)
            print(classes)
            classes = model.test(validation_txt_path_, k=-1, threshold=0.5)
            print(classes)

        else:
            for i in range(1, x_times_average + 1):
                model = fasttext.train_supervised(**_params)

                samples, precision_train, recall_train = model.test(self.train_txt_path, k=-1, threshold=0.5)
                samples, precision_validation, recall_validation = model.test(self.validation_txt_path, k=-1,
                                                                              threshold=0.5)

                precisions_train.append(precision_train)
                recalls_train.append(recall_train)
                precisions_validation.append(precision_validation)
                recalls_validation.append(recall_validation)

            result["precision_train"] = precisions_train
            result["recall_train"] = recalls_train
            result["precision_validation"] = precisions_validation
            result["recall_validation"] = recalls_validation

        return result

    def plot_results(self,
                     results: dict,
                     save_figs_to: str,
                     max_epochs: int,
                     use_pretrained: bool,
                     multipleruns) -> None:
        """
        Function to plot precision, recall and F-score for the trained models.

        :param max_epochs: maximum epoch number for training models
        :param train_precisions: previously calculated prescisions on train set
        :param train_recalls: previously calculated recalls on train set
        :param validation_precisions: previously calculated prescisions on validation set
        :param validation_recalls: previously calculated recalls on validation set
        :param save_figs_to: path to save the created figures
        :return: None
        """

        """
        Internal storage of metrics:
        RESULTS = { 
            epoch_number_1: 
                {
                    "precision_train" :     [min, max, average],
                    "recall_train":         [min, max, average],
                    "precision_validation": [min, max, average],
                    "recall_validation":    [min, max, average],
                },
            epoch_number_2: ...
        }
        """

        # print(results)

        if multipleruns == 0:

            train_precisions = []
            train_recalls = []
            validation_precisions = []
            validation_recalls = []

            for epoch, metrics in results.items():  # epoch_number : { prec_s : [], rec_s: [], ...}
                train_precisions.append(metrics["precision_train"][0])
                train_recalls.append(metrics["recall_train"][0])
                validation_precisions.append(metrics["precision_validation"][0])
                validation_recalls.append(metrics["recall_validation"][0])

            train_p = np.array(train_precisions)
            train_r = np.array(train_recalls)
            train_f1 = np.array(self.__calculate_f1s(train_precisions, train_recalls))
            valid_p = np.array(validation_precisions)
            valid_r = np.array(validation_recalls)
            valid_f1 = np.array(self.__calculate_f1s(validation_precisions, validation_recalls))

            x = np.arange(self.steps_for_epochs, max_epochs + 1, self.steps_for_epochs)

            fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12, 6))

            train_ps, = axs[0].plot(x, train_p, marker='o', color="green")
            valid_ps, = axs[0].plot(x, valid_p, marker='o', color="blue")
            for i in range(len(train_p)):
                axs[0].text(i * self.steps_for_epochs, train_p[i] + 0.01, '{0:.3g}'.format(train_p[i]),
                            horizontalalignment='right', verticalalignment='top')
                axs[0].text(i * self.steps_for_epochs, valid_p[i] - 0.01, '{0:.3g}'.format(valid_p[i]),
                            horizontalalignment='left', verticalalignment='bottom')
            axs[0].legend(handles=[train_ps, valid_ps], labels=["Train P", "Validation P"], loc='lower right')

            train_rs, = axs[1].plot(x, train_r, marker='o', color="green")
            valid_rs, = axs[1].plot(x, valid_r, marker='o', color="blue")
            for i in range(len(valid_p)):
                axs[1].text(i * self.steps_for_epochs, train_r[i] + 0.01, '{0:.3g}'.format(train_r[i]),
                            horizontalalignment='right', verticalalignment='top')
                axs[1].text(i * self.steps_for_epochs, valid_r[i] - 0.01, '{0:.3g}'.format(valid_r[i]),
                            horizontalalignment='left', verticalalignment='bottom')
            axs[1].legend(handles=[train_rs, valid_rs], labels=["Train R", "Validation R"], loc='lower right')

            train_f1s, = axs[2].plot(x, train_f1, marker='o', color="green")
            valid_f1s, = axs[2].plot(x, valid_f1, marker='o', color="blue")
            for i in range(len(train_f1)):
                axs[2].text(i * self.steps_for_epochs, train_f1[i] + 0.01, '{0:.3g}'.format(train_f1[i]),
                            horizontalalignment='right', verticalalignment='top')
                axs[2].text(i * self.steps_for_epochs, valid_f1[i] - 0.01, '{0:.3g}'.format(valid_f1[i]),
                            horizontalalignment='left', verticalalignment='bottom')
            axs[2].legend(handles=[train_f1s, valid_f1s], labels=["Train F1", "Validation F1"], loc='lower right')

            fig.tight_layout(pad=3.0)
            fig.subplots_adjust(top=0.9)
            plt.suptitle(f"Overfitting testing with {str(max_epochs)} epochs", y=0.98)

            axs[0].set_title("Precision vs. Epochs")
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Precision")
            axs[0].grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
            axs[0].grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
            axs[0].minorticks_on()

            axs[1].set_title("Recall vs. Epochs")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Recall")
            axs[1].grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
            axs[1].grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
            axs[1].minorticks_on()

            axs[2].set_title("F1 vs. Epochs")
            axs[2].set_xlabel("Epochs")
            axs[2].set_ylabel("F1")
            axs[2].grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
            axs[2].grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
            axs[2].minorticks_on()

            # get caller function to specify the output figures' name:
            caller = stack()[1].function

            if use_pretrained:
                if caller == "__test_overfitting":
                    plt.savefig(
                        save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps__PRETRAINED_class_test.png")
                else:
                    plt.savefig(
                        save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps__PRETRAINED_earlystop_class_test.png")
            else:
                if caller == "__test_overfitting":
                    plt.savefig(save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps_class_test.png")
                else:
                    plt.savefig(
                        save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps_earlystop_class_test.png")

        else:

            train_p_s = {}
            train_r_s = {}
            valid_p_s = {}
            valid_r_s = {}
            train_p_df_dict = {}
            train_r_df_dict = {}
            valid_p_df_dict = {}
            valid_r_df_dict = {}
            train_f1s_dict = {}
            valid_f1s_dict = {}

            # szerezzük meg a metrikákat
            for epoch, metrics_dict in results.items():
                for metric, values_list in metrics_dict.items():
                    if metric == "precision_train":
                        train_p_s[epoch] = values_list
                    elif metric == "recall_train":
                        train_r_s[epoch] = values_list
                    elif metric == "precision_validation":
                        valid_p_s[epoch] = values_list
                    else:
                        valid_r_s[epoch] = values_list

            for epoch in range(self.steps_for_epochs, max_epochs + 1, self.steps_for_epochs):
                train_p_df_dict[epoch] = self.calc_max_min_q1_q3_mean(train_p_s[epoch])
                train_r_df_dict[epoch] = self.calc_max_min_q1_q3_mean(train_r_s[epoch])
                valid_p_df_dict[epoch] = self.calc_max_min_q1_q3_mean(valid_p_s[epoch])
                valid_r_df_dict[epoch] = self.calc_max_min_q1_q3_mean(valid_r_s[epoch])
                train_f1_list = self.__calculate_f1s(train_p_s[epoch], train_r_s[epoch])
                valid_f1_list = self.__calculate_f1s(valid_p_s[epoch], valid_r_s[epoch])
                train_f1s_dict[epoch] = self.calc_max_min_q1_q3_mean(train_f1_list)
                valid_f1s_dict[epoch] = self.calc_max_min_q1_q3_mean(valid_f1_list)

            filename = ""

            caller = stack()[1].function

            if use_pretrained:
                if caller == "__test_overfitting":
                    filename = save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps__PRETRAINED_class_test.png"
                else:
                    filename = save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps__PRETRAINED_earlystop_class_test.png"
            else:
                if caller == "__test_overfitting":
                    filename = save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps_class_test.png"
                else:
                    filename = save_figs_to + f"/{max_epochs}_epoch_{self.steps_for_epochs}_steps_earlystop_class_test.png"

            self.save_fig(train_p_df_dict, ["Epochs", "Train Precision"], filename.replace(".png", "_TrainP.png"))
            self.save_fig(train_r_df_dict, ["Epochs", "Train Recall"], filename.replace(".png", "_TrainR.png"))
            self.save_fig(valid_p_df_dict, ["Epochs", "Validation Precision"], filename.replace(".png", "_ValidP.png"))
            self.save_fig(valid_r_df_dict, ["Epochs", "Validation Recall"], filename.replace(".png", "_ValidR.png"))
            self.save_fig(train_f1s_dict, ["Epochs", "Train F1"], filename.replace(".png", "_TrainF1.png"))
            self.save_fig(valid_f1s_dict, ["Epochs", "Validation F1"], filename.replace(".png", "_ValidF1.png"))

    def calc_max_min_q1_q3_mean(self,
                                data_list: list
                                ) -> list:

        q1 = np.quantile(data_list, .25)
        q3 = np.quantile(data_list, .75)
        min_ = min(data_list)
        max_ = max(data_list)
        mean = sum(data_list) / len(data_list)
        return [max_, min_, q1, q3, mean]

    def save_fig(self,
                 data_dict: Dict,
                 axis_labels: list,
                 save_fig_to: str
                 ) -> None:

        # print(data_dict)
        df_from_data = pd.DataFrame(data_dict)
        df_from_data.index = ['max', 'min', 'q1', 'q3', 'med']
        # print(df_from_data)

        labels = list(df_from_data.columns)  # epochs

        bxp_stats = df_from_data.apply(
            lambda x: {'med': x.med, 'q1': x.q1, 'q3': x.q3, 'whislo': x['min'], 'whishi': x['max']},
            axis=0).tolist()

        for index, item in enumerate(bxp_stats):
            item.update({'label': labels[index]})

        _, ax = plt.subplots()
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.bxp(bxp_stats, showfliers=False)
        plt.savefig(save_fig_to)
        # plt.show()

    def __calculate_f1s(self, test_p: list, test_r: list) -> list:
        """
        Calculates and returns F-Scores from lists of Precisions and Recalls.

        :param test_p: List of precisions
        :param test_r: List of corresponding Recalls
        :return:
        """

        f1s = []
        for p, r in zip(test_p, test_r):
            f1s.append(self.__calc_f1(p, r))
        return f1s

    def __calc_f1(self, p, r):
        """
        Calculates F-Score from precision and recall given as a parameter

        :param p: precision
        :param r: recall
        :return:
        """

        return 2 * p * r / (p + r)


if __name__ == '__main__':
    train_txt_path_ = "/home/istvanu/WK/test/data/train_validation/kozig_fasttext_lower_train.txt"
    validation_txt_path_ = "/home/istvanu/WK/test/data/train_validation/kozig_fasttext_lower_validation.txt"
    save_figs_to_ = "/home/istvanu/PycharmProjects/wk-2022-1/tests/test_resources"
    steps_for_epochs = 5
    max_epochs_number = 50

    params = {"input": train_txt_path_,
              "lr": 0.7,
              "epoch": max_epochs_number,
              "wordNgrams": 2,
              "bucket": 200000,
              "dim": 50,
              "loss": 'ova',
              "minCount": 10,
              "seed": 1,
              "pretrainedVectors": "/home/istvanu/WK/fastText_models/cc.hu.50.vec",
              "neg": 20
              }

    predictor = OverfitPredictor(train_txt_path_,
                                 validation_txt_path_,
                                 save_figs_to_,
                                 steps_for_epochs,
                                 use_pretrained_=True
                                 # set true, IF "pretrainedVectors" defined in params (comment otherwise)
                                 )

    ########################################## start without early stopping ###########################################
    predictor.start(
        params,
        # randomseed=True,        # optional
        multipleruns=5  # optional
    )

    ############################################ start with early stopping ############################################
    # predictor.start_with_earlystopping(
    #     params,
    #     monitor="precision",
    #     patience=2,             # optional
    #     min_delta=0.05,         # optional
    #     randomseed=False       # optional
    # )

    ############################################ testing ############################################

    # test_dict = {1: {'precision_train': [0.541684440020854, 0.6416798489564729, 0.6140775300961926, 0.7434897554527429, 0.6571613320400905],
    #                  'recall_train': [0.3669374257552894, 0.40373069637525283, 0.3422801553921726, 0.3611583780139339, 0.32629145664108905],
    #                  'precision_validation': [0.5305429864253394, 0.6428717529147064, 0.6106032906764168, 0.7449521785334751, 0.6506211180124224],
    #                  'recall_validation': [0.36295627499032634, 0.4053914613697923, 0.3446407842125629, 0.3616664516961176, 0.32426157616406553]},
    #              2: {'precision_train': [0.7123416678438574, 0.7759055982436882, 0.8441399059112293, 0.810727072431837, 0.7829157929588345],
    #                  'recall_train': [0.48569685684014513, 0.4538799884419045, 0.397502167142903, 0.3780460397470061, 0.45551738530195524],
    #                  'precision_validation': [0.7027939246202888, 0.7716089701720009, 0.8443654266958425, 0.800382304751502, 0.7793664760228772],
    #                  'recall_validation': [0.4834257706694183, 0.4571133754675609, 0.39816845092222364, 0.37804720753256804, 0.4569843931381401]},
    #              3: {'precision_train': [0.8553860992223763, 0.8055335968379447, 0.8215922593787133, 0.7655886704713591, 0.7832849245760329],
    #                  'recall_train': [0.45558159694352585, 0.49073747070343854, 0.4661765178026776, 0.4651491315375478, 0.536809323530356],
    #                  'precision_validation': [0.8502686858817782, 0.8040455120101138, 0.8197952218430035, 0.7585702341137124, 0.7733158191030212],
    #                  'recall_validation': [0.4489874887140462, 0.4921965690700374, 0.46472333290339224, 0.4680768734683348, 0.5315361795434026]}}
    #
    # # https://stackoverflow.com/questions/62125848/matplotlib-dataframe-boxplot-with-given-max-min-and-quaritles
    #
    # predictor.plot_results(test_dict,
    #                        save_figs_to=save_figs_to_,
    #                        max_epochs=3,
    #                        use_pretrained=True,
    #                        multipleruns=3)

    # data ={1: {'precision_train': [0.605258180442595, 0.605258180442595, 0.605258180442595], 'recall_train': [0.4109545060519472, 0.4109545060519472, 0.4109545060519472], 'precision_validation': [0.6019563581640331, 0.6019563581640331, 0.6019563581640331], 'recall_validation': [0.4127434541467819, 0.4127434541467819, 0.4127434541467819]}, 2: {'precision_train': [0.711997752177578, 0.711997752177578, 0.711997752177578], 'recall_train': [0.48813689921982856, 0.48813689921982856, 0.48813689921982856], 'precision_validation': [0.7020720552548068, 0.7020720552548068, 0.7020720552548068], 'recall_validation': [0.4851025409518896, 0.4851025409518896, 0.4851025409518896]}, 3: {'precision_train': [0.7851302417941829, 0.7851302417941829, 0.7851302417941829], 'recall_train': [0.4315985488169005, 0.4315985488169005, 0.4315985488169005], 'precision_validation': [0.7774654485828063, 0.7774654485828063, 0.7774654485828063], 'recall_validation': [0.42809235134786533, 0.42809235134786533, 0.42809235134786533]}, 4: {'precision_train': [0.8670562101603245, 0.8670562101603245, 0.8670562101603245], 'recall_train': [0.5764600122002119, 0.5764600122002119, 0.5764600122002119], 'precision_validation': [0.8615591397849462, 0.8615591397849462, 0.8615591397849462], 'recall_validation': [0.5787437121114407, 0.5787437121114407, 0.5787437121114407]}, 5: {'precision_train': [0.8770149778187728, 0.8770149778187728, 0.8770149778187728], 'recall_train': [0.6410569236202524, 0.6410569236202524, 0.6410569236202524], 'precision_validation': [0.8699630476860813, 0.8699630476860813, 0.8699630476860813], 'recall_validation': [0.637688636656778, 0.637688636656778, 0.637688636656778]}, 6: {'precision_train': [0.911324273914202, 0.911324273914202, 0.911324273914202], 'recall_train': [0.5490416412495586, 0.5490416412495586, 0.5490416412495586], 'precision_validation': [0.9025946405784773, 0.9025946405784773, 0.9025946405784773], 'recall_validation': [0.5474010060621695, 0.5474010060621695, 0.5474010060621695]}, 7: {'precision_train': [0.8705653778900331, 0.8705653778900331, 0.8705653778900331], 'recall_train': [0.7192988088740488, 0.7192988088740488, 0.7192988088740488], 'precision_validation': [0.857567229518449, 0.857567229518449, 0.857567229518449], 'recall_validation': [0.7074680768734684, 0.7074680768734684, 0.7074680768734684]}}
    # predictor.plot_results(data,
    #                        save_figs_to=save_figs_to_,
    #                        max_epochs=7,
    #                        use_pretrained=True,
    #                        multipleruns=0)
