# Deep Learning Course Labs

Кураторский репозиторий с лабораторными работами по глубокому обучению: от ручного forward pass и настройки обучения нейросетей до transfer learning, reinforcement learning и GAN.

В репозиторий включены только наиболее содержательные notebooks без сохранённых runtime errors и без явно незавершённых шаблонов.

> Лабораторные выполнены в рамках университетского курса по глубокому обучению на основе учебных материалов преподавателя Н. В. Блохина. Репозиторий показывает выполненные упражнения и эксперименты; авторство исходных учебных материалов преподавателю не приписывается студенту.

## Содержание

| Notebook | Тема | Что демонстрирует |
|---|---|---|
| `01_forward_pass.ipynb` | Forward pass | полносвязные слои, функции активации, softmax, функции потерь и ручное прохождение данных через сеть |
| `02_training_regularization.ipynb` | Training & regularization | Xavier/Kaiming initialization, SGD/RMSprop/Adam, Dropout, BatchNorm и сравнение поведения моделей |
| `03_transfer_learning_cnn.ipynb` | Transfer learning | использование pretrained VGG16, заморозка/дообучение слоёв, ImageFolder/DataLoader и оценка accuracy |
| `04_policy_gradients.ipynb` | Deep Reinforcement Learning | policy-gradient подход в Gymnasium/CartPole, нейронная policy, discounted rewards и обучение по траекториям |
| `05_gan.ipynb` | Generative Deep Learning | Generator/Discriminator, сверточные и transposed-convolution слои, генерация изображений |

## 1. Forward Pass

Базовая лабораторная по внутренней механике нейронной сети:

- создание полносвязных преобразований;
- применение функций активации;
- softmax для многоклассового выхода;
- вычисление функции потерь;
- проверка операций на PyTorch tensors.

Notebook полезен как демонстрация понимания того, что происходит внутри `nn.Module`, а не только использования готовых моделей.

## 2. Инициализация, оптимизация и регуляризация

Лабораторная посвящена устойчивому обучению нейронных сетей:

- Xavier initialization;
- Kaiming initialization;
- SGD;
- RMSprop;
- Adam;
- Dropout;
- Batch Normalization;
- анализ различий между train/test поведением.

![Training and regularization](assets/training_regularization_1.png)

## 3. Transfer Learning для классификации изображений

Работа с pretrained CNN:

- загрузка изображений через `ImageFolder`;
- подготовка `DataLoader`;
- использование `VGG16`;
- заморозка и настройка параметров;
- fine-tuning;
- отслеживание loss и accuracy на train/test.

В сохранённом эксперименте лучшая accuracy находилась примерно в диапазоне `0.8+`, что используется как учебный пример сравнения стратегий переноса обучения.

![Transfer learning](assets/transfer_learning_1.png)

## 4. Policy Gradients

Deep Reinforcement Learning на среде `CartPole`:

- нейронная сеть задаёт policy;
- действия семплируются из распределения вероятностей;
- сохраняются log-probabilities;
- рассчитывается discounted return;
- policy обновляется по собранным эпизодам;
- анализируется динамика reward.

![Policy gradients rewards](assets/policy_gradients_1.png)

## 5. GAN

Лабораторная по генеративно-состязательным сетям:

- отдельные сети `Generator` и `Discriminator`;
- сверточные слои;
- `ConvTranspose2d` для генератора;
- обучение на наборе изображений;
- визуализация генерируемых примеров.

![GAN output](assets/gan_1.png)

## Почему не включены остальные notebooks

При отборе были просмотрены соседние лабораторные из папок курса `01_intro`, `02_nn`, `03_adv`, `04_cnn`, `05_rl`, `06_cv`.

Не включены:

- слишком базовые вводные notebooks по sklearn/tensors/data;
- `02_3_nn_backprop.ipynb` и `02_4_torch_nn.ipynb` — в сохранённых outputs присутствуют предупреждения о несовпадении размерностей target/input;
- `04_3_cnn_1d.ipynb` — сохранён `RuntimeError`;
- `05_3_dqn.ipynb` — часть итогового Trainer оставлена как шаблон с `pass` / `...`;
- `06_4_lightening_deploy.ipynb` — сохранён `FileNotFoundError`;
- object detection и image segmentation не дублируются здесь, поскольку они лучше подходят для отдельного Computer Vision репозитория.

## Структура

```text
deep-learning-labs/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
└── notebooks/
    ├── 01_forward_pass.ipynb
    ├── 02_training_regularization.ipynb
    ├── 03_transfer_learning_cnn.ipynb
    ├── 04_policy_gradients.ipynb
    └── 05_gan.ipynb
```

## Установка

```bash
pip install -r requirements.txt
```

Часть notebook использует старые версии библиотек и Google Colab, поэтому при полном повторном запуске отдельных ячеек может потребоваться небольшая адаптация API.

## Стек

`Python` · `PyTorch` · `torchvision` · `NumPy` · `pandas` · `scikit-learn` · `Matplotlib` · `Gymnasium`
