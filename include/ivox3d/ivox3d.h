//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <execution>
#include <list>
#include <thread>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace faster_lio {

enum class IVoxNodeType {
    DEFAULT,  // linear ivox
    PHC,      // phc ivox
};

/// traits for NodeType
template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits {};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
    using NodeType = IVoxNode<PointT, dim>;
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim> {
    using NodeType = IVoxNodePhc<PointT, dim>;
};

template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZ>
class IVox {
   public:
//    体素的key值类型是一个3维向量
    using KeyType = Eigen::Matrix<int, dim, 1>;
    // 体素内的点的类型也是存储为一个3维向量
    using PtType = Eigen::Matrix<float, dim, 1>;
    using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim>::NodeType;
    // 体素内的点集类型
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    // 用户决定的邻近体素个数
    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    // 存储体素的参数
    struct Options {
        float resolution_ = 0.2;                        // ivox resolution
        float inv_resolution_ = 10.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    /**
     * constructor
     * @param options  ivox options
     */
    // 初始化一个体素后会生成近邻体素
    explicit IVox(Options options) : options_(options) {
        // 配置体素的分辨率
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }

    /**
     * add points
     * @param points_to_add
     */
    void AddPoints(const PointVector& points_to_add);

    /// get nn
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt);

    /// get nn with condition
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0);

    /// get nn in cloud
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);

    /// get number of points
    size_t NumPoints() const;

    /// get number of valid grids
    size_t NumValidGrids() const;

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

   private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    // KeyType = Eigen::Matrix<int, dim, 1>
    KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    // 注意typename std::list<std::pair<KeyType, NodeType>>::iterator这里的iterator，那么这个值指list中的一个元素，而不是整个list
    // 整个list的元素构成整个体素地图
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        grids_map_;                                        // voxel hash map
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache
    std::vector<KeyType> nearby_grids_;                    // nearbys
};

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointType& closest_pt) {
    std::vector<DistPoint> candidates;
    auto key = Pos2Grid(ToEigen<float, dim>(pt));
    std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&key, &candidates, &pt, this](const KeyType& delta) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) {
                candidates.emplace_back(dist_point);
            }
        }
    });

    if (candidates.empty()) {
        return false;
    }

    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num,
                                                      double max_range) {
    // 候选近邻点，每个点都有对应的距离
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());

    // 计算该点所属体素的索引
    auto key = Pos2Grid(ToEigen<float, dim>(pt));

// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["knn"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif

    // 遍历近邻的体素， nearby_grids_存储的是邻近体素的相对偏移，
    // 将相对偏移叠加上当前体素的绝对索引，即可得到邻近体素的绝对索引
    for (const KeyType& delta : nearby_grids_) {
        // 在地图中的实际体素索引
        auto dkey = key + delta;
        // 找到对应的体素
        auto iter = grids_map_.find(dkey);
        // 如果可以找到对应的体素，则进行KNN最近邻搜索
        if (iter != grids_map_.end()) {
#ifdef INNER_TIMER
            auto t1 = std::chrono::high_resolution_clock::now();
#endif
            auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
#ifdef INNER_TIMER
            auto t2 = std::chrono::high_resolution_clock::now();
            auto knn = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            stats["knn"].emplace_back(knn);
#endif
        }
    }

    // 如果候选点队列为空，则返回false，代表找不到近邻点
    if (candidates.empty()) {
        return false;
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    if (candidates.size() <= max_num) {
    } else { // 如果候选点大于max_num(5)个，则需要根据距离进行选择
        // nth_element(temp.begin(), temp.begin()+10,temp.end()); 
        // 该函数的作用为将迭代器指向的从_First 到 _last 之间的元素进行二分排序，
        // 以10为分界，前面都比10小（大），后面都比之大（小）；但是两段内并不是有序的，
        // 特别适用于找出前k个最大（最小）的元素。
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        // 这里的意思是：从candidates中找到距离排第max_num点并将其放在candidates[max_num - 1]，
        //              并将距离小于它的放在前半段，反之则放在后半段
        candidates.resize(max_num);
        // 直接通过resize截除后半段距离太远的点
    }
    // 进一步，将距离最小的点挪到第一位
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    // 将近邻点放进去closest_pt
    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumValidGrids() const {
    return grids_map_.size();
}

// 为当前体素生成近邻体素坐标的相对偏移量
template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GenerateNearbyGrids() {
    // 参数文件中 ivox_nearby_type: 18   # 6, 18, 26
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
                         KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else {
        LOG(ERROR) << "Unknown nearby_type!";
    }
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    // 给点云的每个点分配索引
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        index[i] = i;
    }
    closest_cloud.resize(cloud.size());

    // 多线程遍历点云
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt)) {
            closest_cloud[idx] = pt;
        } else {
            closest_cloud[idx] = PointType();
        }
    });
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::AddPoints(const PointVector& points_to_add) {
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        // 将点的坐标转成hash值索引(3维的)
        auto key = Pos2Grid(ToEigen<float, dim>(pt));

        // 在栅格地图中寻找是否有对应的栅格
        auto iter = grids_map_.find(key); // iter->second->second : NodeType/IVoxNode
        // 没有对应的栅格的话就要新建栅格
        if (iter == grids_map_.end()) {
            PointType center;
            // getVector3fMap() : pcl::PointXYZ将转成Eigen::Vector3f
            // 计算新的体素中心坐标，也就是新加入的点的坐标
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;

            // 在栅格地图缓存中加入新的体素
            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            // 让key和对应体素绑定
            grids_map_.insert({key, grids_cache_.begin()});

            // 向该体素中加入点
            grids_cache_.front().second.InsertPoint(pt);

            // 如果地图的总体素个数大于阈值，则要将旧的体素删除，如论文Fig.4
            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        // 寻找到已有栅格
        } else {
            // 每个体素里通过vector存放点，InsertPoint就是emplace_back
            iter->second->second.InsertPoint(pt);
            // 缓存拼接, https://blog.csdn.net/boiled_water123/article/details/103753598中的方法二
            // 将iter->second剪接到grids_cache_.begin()的位置，维护体素地图中的体素由新到旧进行排列
            // 对应论文中的Fig.4
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            // 重新让key和对应体素绑定
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType& pt) const {
    // .array() 方法转成Array类，之后才可以使用round()
//         Eigen 不仅提供了Matrix和Vector结构，还提供了Array结构。
//         区别如下，Matrix和Vector就是线性代数中定义的矩阵和向量，所
//         有的数学运算都和数学上一致。但是存在一个问题是数学上的定义并
//         不一定能完全满足现实需求。比如，数学上并没有定义一个矩阵和一
//         个标量的加法运算。但是如果我们想给一个矩阵的每个元素都加上同
//         一个数，那么这个操作就需要我们自己去实现，这显然并不方便。

// 　　    Array提供了一个Array类，为我们提供了大量的矩阵未定义的操作，
//         且Array和Matrix之间很容易相互转换，所以相当于给矩阵提供更多
//         的方法。也为使用者的不同需求提供了更多的选择。
    // round() 方法返回浮点数x的四舍五入值
    // pt的坐标除以体素分辨率，然后取整型
    return (pt * options_.inv_resolution_).array().round().template cast<int>();
}

template <int dim, IVoxNodeType node_type, typename PointType>
std::vector<float> IVox<dim, node_type, PointType>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }
    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
    return std::vector<float>{valid_num, ave, max, min, stddev};
}

}  // namespace faster_lio

#endif
