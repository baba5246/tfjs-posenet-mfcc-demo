// MFCC variables
// setup init variables
const FEATURE_NAME_MFCC = 'mfcc'
const FEATURE_NAME_RMS = 'rms'
const NUM_MFCC_COEF = 20
const THRESHOLD_RMS = 0.002 // threshold on rms value
const MFCC_HISTORY_MAX_LENGTH = 20
let curMFCC = null
let curRMS = 0
let mfccHistory = []
let mfccFeatures = []

// posenet settings
const imageScaleFactor = 0.2
const outputStride = 16
const flipHorizontal = true
const contentWidth = 600
const contentHeight = 480
let curKeypoints = null

// training settings
let trainData = []
const NUM_FRAMES = MFCC_HISTORY_MAX_LENGTH
const INPUT_SHAPE = [NUM_FRAMES, NUM_MFCC_COEF, 1]
let model = null

let audioStream = null
let videoStream = null
const canvas = document.getElementById('video_canvas')
const micStartBtn = document.getElementById('mic_start_btn')
const recordBtn = document.getElementById('record_btn')
const trainBtn = document.getElementById('train_btn')
const listenBtn = document.getElementById('listen_btn')

let recordClearId = null

window.addEventListener('load', () => {
  let constraints = {audio: true, video: true}
  streams = navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    audioStream = new MediaStream(stream.getAudioTracks())
    videoStream = new MediaStream(stream.getVideoTracks())

    // video load -> posenet
    const video = document.getElementById('video')
    video.srcObject = videoStream
    video.onloadedmetadata = () => {
      video.play()
      posenet.load().then((net) => {
        detectPoseInRealTime(video, net)
      })
    }
  })
})

function detectPoseInRealTime(video, net) {
  const ctx = canvas.getContext('2d')

  async function poseDetectionFrame() {
    let poses = []
    const pose = await net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride)
    poses.push(pose)

    ctx.clearRect(0, 0, contentWidth, contentHeight)

    ctx.save()
    ctx.drawImage(video, 0, 0, contentWidth, contentHeight)
    ctx.restore()

    poses.forEach(({score, keypoints}) => {
      curKeypoints = keypoints
      .slice(0, 5)
      .map(kp => [
        kp.position.x / contentWidth,
        kp.position.y / contentHeight
      ])
      drawPoint(keypoints[0], ctx)
      drawPoint(keypoints[1], ctx)
      drawPoint(keypoints[2], ctx)
      drawPoint(keypoints[3], ctx)
      drawPoint(keypoints[4], ctx)
    })

    requestAnimationFrame(poseDetectionFrame)
  }

  poseDetectionFrame()
}

function drawPoint(kp, ctx, color = 'pink') {
  ctx.beginPath()
  ctx.arc(kp.position.x, kp.position.y, 3, 0, 2 * Math.PI)
  ctx.fillStyle = color
  ctx.fill()
}

micStartBtn.onclick = () => {
  const AudioContext = window.AudioContext || window.webkitAudioContext
  const audioCtx = new AudioContext()

  const audioSrc = audioCtx.createMediaStreamSource(audioStream)
  const features = [FEATURE_NAME_MFCC, FEATURE_NAME_RMS]
  Meyda.createMeydaAnalyzer({
    'audioContext': audioCtx,
    'source': audioSrc,
    'bufferSize': 1024,
    'featureExtractors': features,
    'numberOfMFCCCoefficients': NUM_MFCC_COEF,
    'callback': saveAudioFeatures
  })
  .start()
}

function saveAudioFeatures(features) {
  curMFCC = features[FEATURE_NAME_MFCC]
  curRMS = features[FEATURE_NAME_RMS]
  mfccHistory.push(curMFCC)
  if (mfccHistory.length > MFCC_HISTORY_MAX_LENGTH) {
    mfccHistory.splice(0, 1)
  }
  const mfccValues = mfccHistory.flat()
  const mean = math.mean(mfccValues)
  const std = math.std(mfccValues)
  mfccFeatures = mfccHistory.map(mfcc => mfcc.map(v => (v - mean) / std))
}

recordBtn.onmousedown = () => {
  recordClearId = setInterval(() => {
    trainData.push({
      mfcc: mfccFeatures,
      kp: curKeypoints
    })
    console.log(trainData.length)
  }, 10)
}

recordBtn.onmouseup = () => {
  clearInterval(recordClearId)
}

trainBtn.onclick = async () => {
  model = tf.sequential()
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 20,
    kernelSize: [10, 10],
    activation: 'relu',
    inputShape: INPUT_SHAPE
  }))
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 1]}))
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 10,
    kernelSize: [7, 7],
    activation: 'relu',
    inputShape: INPUT_SHAPE
  }))
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 1]}))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({units: 512}))
  model.add(tf.layers.dropout({rate: 0.30}))
  model.add(tf.layers.dense({units: 32}))
  model.add(tf.layers.dense({units: 10}))
  const optimizer = tf.train.adam(0.01)
  model.compile({
    optimizer,
    loss: 'meanSquaredError',
    metrics: ['mse', 'mae', 'mape', 'cosine']
  })
  console.log(model.summary())

  trainData = shuffle(trainData)

  const ys = tf.tensor(trainData.map(d => d.kp.flat()))
  const xsShape = [trainData.length, ...INPUT_SHAPE]
  const xs = tf.tensor(trainData.map(d => d.mfcc), xsShape)

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`MSE: ${logs.mse.toFixed(4)} % Epoch: ${epoch + 1}`)
      }
    }
  })
  tf.dispose([xs, ys])
}

listenBtn.onclick = () => {
  setInterval(() => {
    const xs = tf.tensor([mfccFeatures], [1, ...INPUT_SHAPE])
    const results = model.predict(xs).dataSync()
    drawPredicted(results)
  }, 10)
}


function drawPredicted(points) {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, contentWidth, contentHeight)
  ctx.save()
  ctx.drawImage(video, 0, 0, contentWidth, contentHeight)
  ctx.restore()
  const keypoints = []
  for (let i = 0; i < points.length; i += 2) {
    keypoints.push({
      position: {
        x: points[i] * contentWidth,
        y: points[i + 1] * contentHeight
      }
    })
  }
  keypoints.forEach((kp) => {
    drawPoint(kp, ctx, 'red')
  })
}

const shuffle = (array) => {
  for (let i = array.length - 1; i >= 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

