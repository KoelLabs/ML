<img width="100%" alt="Koel Lab's Logomark" src="https://github.com/user-attachments/assets/0e7cf381-3c44-402e-a58d-4af419037a1e" />

[![Mozilla Builders](https://img.shields.io/badge/Mozilla-000000.svg?style=for-the-badge&logo=Mozilla&logoColor=white)](https://future.mozilla.org/builders/)
![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)
![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)

# Koel Labs - Machine Learning
![Black Formatting](https://github.com/KoelLabs/ML/actions/workflows/black.yml/badge.svg)
![Zizmor](https://github.com/KoelLabs/ML/actions/workflows/zizmor.yml/badge.svg)
![Gitleaks Secret Scanning](https://github.com/KoelLabs/ML/actions/workflows/gitleaks.yml/badge.svg)

Contains the EDA, training, evaluation, and data processing code for Koel Labs. Evaluation results will be made available via [Hugging Face Leaderboards](https://huggingface.co/spaces/KoelLabs/IPA-Transcription-EN). Cleaned datasets and model weights will also be made available via [Hugging Face](https://huggingface.co/KoelLabs). We will be releasing a paper as well so stay tuned!

Read about all our repositories [here](https://github.com/KoelLabs).

## Setup

Checkout the [guides](./guides) directory for standalone guides on finetuning, evaluation, dataset processing, and other topics. 
These can be run independently of the setup for the rest of the codebase, e.g., in a Colab notebook.

See the [DEVELOPMENT.md](DEVELOPMENT.md) for alternative setup instructions and details.

0. `git clone https://github.com/KoelLabs/ML.git`
1. Install Python 3.8.10
2. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
3. Run the commands in './scripts/install.sh', e.g., with `. ./scripts/install.sh`.

## Contributing

Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for specific guidelines on contributing to this repository.

## License

The code in this repository is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

With the exception of a few models and Huggingface spaces released during the builders program under the [Mozilla Public License](https://www.mozilla.org/en-US/MPL/2.0/), all Huggingface models and code will be released under the GNU Affero General Public License.

We retain all rights to the Koel Labs brand, logos, blog posts and website content.
