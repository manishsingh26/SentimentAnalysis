

import io
import csv
from textblob import TextBlob
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
import Features, SentimentOptions


class SentimentAnalysis(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text_array = []
        self.csv_out = csv.writer(open("SentimentalAnalysisNetworkPerformance.csv", "w", newline=""))
        self.header = ["Algorithm", "Comment", "Key1", "Val1", "Key2", "Val2", "Key3", "Val3", "Key4", "Val4"]
        self.csv_out.writerow(self.header)

    def read_pdf(self):
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(self.pdf_path, "rb") as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()

            for each_line in text.split(""):

                if "Additional Comments" in str(each_line):
                    each_line_edited = each_line.replace("Additional Comments", "").strip()
                    ignore_array = ["NA", "na", "none", "NONE"]

                    if each_line_edited not in ignore_array:
                        self.text_array.append(each_line_edited)

    def text_blob_main(self):
        polarity_array = []
        subjectivity_array = []

        blob_output = TextBlob(",".join(self.text_array))

        for each_line in self.text_array:
            blob_output = TextBlob(each_line)
            data_array = ["Text Blob", each_line, "polarity", blob_output.sentiment.polarity, "subjectivity",
                          blob_output.sentiment.subjectivity, "assessments",
                          blob_output.sentiment_assessments.assessments]
            self.csv_out.writerow(data_array)
            if sentence.polarity != 0 and sentence.subjectivity != 0:
                polarity_array.append(sentence.polarity)
                subjectivity_array.append(sentence.subjectivity)

        polarity_mean = sum(polarity_array)/len(polarity_array)
        subjectivity_mean = sum(subjectivity_array)/len(subjectivity_array)

        print("Text Blob Algorithm ::")
        print("Polarity Mean", polarity_mean)
        print("Subjectivity Mean", subjectivity_mean)

    def vader_main(self):
        negative_array = []
        neutral_array = []
        positive_array = []
        compound_array = []

        for each_line in self.text_array:
            vader_output = SentimentIntensityAnalyzer()
            vader_data = vader_output.polarity_scores(each_line)
            data_array = ["Vader", each_line, "Negative", vader_data["neg"], "Neutral", vader_data["neg"], "Positive",
                          vader_data["pos"], "Compound", vader_data["compound"]]
            self.csv_out.writerow(data_array)
            negative_array.append(vader_data["neg"])
            neutral_array.append(vader_data["neu"])
            positive_array.append(vader_data["neu"])
            compound_array.append(vader_data["compound"])

        negative_mean = sum(negative_array) / len(negative_array)
        neutral_mean = sum(neutral_array) / len(neutral_array)
        positive_mean = sum(positive_array) / len(positive_array)
        compound_mean = sum(compound_array) / len(compound_array)

        print("Vader Algorithm ::")
        print("Negative Mean", negative_mean)
        print("Neutral Mean", neutral_mean)
        print("Positive Mean", positive_mean)
        print("Compound Mean", compound_mean)

    def ibm_watson_main(self):

        for each_line in self.text_array:
            print(each_line)
            natural_language_understanding = NaturalLanguageUnderstandingV1(version='2018-11-16',
                                                                            iam_apikey='w7GrxJ3UIFohkQfRhoWV57Oup_g6T_ZzaykROswSNaKD',
                                                                            url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')

            response = natural_language_understanding.analyze(text=each_line, features=Features(emotion=EmotionOptions(), sentiment=SentimentOptions())).get_result()
            print(response)
            natural_language_understanding = NaturalLanguageUnderstandingV1(
                username='manishsingh2687@gmail.com',
                password='HA609jot159$',
                version='2019-12-23')
            response = natural_language_understanding.analyze(text=each_line, features=Features(sentiment=SentimentOptions())).get_result()
            print(json.dumps(response, indent=2))
            response = natural_language_understanding.analyze(url='www.wsj.com/news/markets', features=Features(sentiment=SentimentOptions()))
            print(response)


if __name__ == "__main__":

    pdf_path_ = r"C:\Users\m4singh\PycharmProjects\NetworkPerformanceSentimentAnalysis\InputData\NWPerformance.pdf"

    obj = SentimentAnalysis(pdf_path_)
    obj.read_pdf()
    obj.text_blob_main()
    obj.vader_main()
    obj.ibm_watson_main()

