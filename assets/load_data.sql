-- Training input for DelayModel.preprocess()
SELECT
  `Fecha-I`,
  `Fecha-O`,
  OPERA,
  TIPOVUELO,
  MES
FROM `{project}.{dataset}.{raw_table}`
