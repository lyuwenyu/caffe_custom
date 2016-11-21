template <typename Dtype>
int ImageDataLayer<Dtype>::myrandomdata (int i) { return caffe_rng_rand()%i;}


template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  /*caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);*/


  const int num_images = lines_.size();
  DLOG(INFO) << "My Shuffle.";
  vector<std::pair<std::string, int> > tlines_;
  vector<int> tnum;
  int pairsize = this->layer_param_.image_data_param().pair_size();

  for(int i = 0; i < num_images / pairsize; i ++)
  {
	  tnum.push_back(i);
  }
  std::random_shuffle(tnum.begin(), tnum.end(), ImageDataLayer<Dtype>::myrandomdata);
  tlines_.clear();
  for(int i = 0; i < num_images / pairsize; i ++)
  {
	  for(int j = 0; j < pairsize; j ++)
	  {
		  tlines_.push_back(lines_[tnum[i] * pairsize + j]);
	  }
  }
  lines_ = tlines_;


}
