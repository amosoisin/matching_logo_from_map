import cv2

class AKAZE: 
    th_val = 100
    
    def __init__(self, img_logo, img_map):
        self.img_logo = img_logo
        self.img_map = img_map
        self.aspect_ratio = img_logo.shape[0]/img_logo.shape[1] # w/h
        self.extend_size = 100
        self.IMG_SIZE = (self.extend_size, int(self.aspect_ratio*self.extend_size))
        self.akaze = cv2.AKAZE_create()
        self.match_num = 0

    def pick_up_logo(self, img):
        img_logo_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_logo_bw = cv2.threshold(img_logo_gray, 150, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(img_logo_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        logo_area = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x:x[1])[-2][0]
        x, y, w, h = cv2.boundingRect(logo_area)
        logo_area = img[y:y+h, x:x+w]
        cv2.imwrite("logo_area.png", logo_area)
        return logo_area

    def full_matching(self):
        img_logo_gray = cv2.cvtColor(self.img_logo, cv2.COLOR_BGR2GRAY)
        img_logo_gray = cv2.resize(img_logo_gray, self.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        img_logo_bw = cv2.threshold(img_logo_gray, self.th_val, 255, cv2.THRESH_BINARY)[1]

        img_map_gray = cv2.cvtColor(self.img_map, cv2.COLOR_BGR2GRAY)
        img_map_bw = cv2.threshold(img_map_gray, self.th_val, 255, cv2.THRESH_BINARY)[1]

        distance_list = []
        image_list = []
        contours, hierarchy = cv2.findContours(img_map_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        idx = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1e2 or 1e5 < area or len(c) <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if self.aspect_ratio-0.3 < w/h < self.aspect_ratio+0.3:
                result_gray = img_map_gray[y:y+h, x:x+w]
                result = self.img_map[y:y+h, x:x+w]
                if self.matching(img_logo_gray, result_gray):
                    distance = self.color_matching(self.img_logo, result)
                    image_list.append((x, y, w, h))
                    distance_list.append((idx, distance))
                    idx += 1

        result_idx = sorted(distance_list, key=lambda x:x[1])[-1][0]
        return image_list[result_idx]

    def color_matching(self, img1, img2):
        img1 = cv2.resize(img1, self.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        img2 = cv2.resize(img2, self.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
        img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
        val = cv2.compareHist(img1_hist, img2_hist, 0)
        return val

    def matching(self, img1, img2):
        img2 = cv2.resize(img2, self.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        kp1, dest1 = self.akaze.detectAndCompute(img1, None)
        kp2, dest2 = self.akaze.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        if len(kp1) <= 1 or len(kp2) <= 1:
            return None
        matches = bf.knnMatch(dest1, dest2, k=2)
        ret = 0.7
        matches = sorted([m for m in matches if m[0].distance < ret*m[1].distance], key=lambda x:x[0].distance)
        if len(matches) == 0:
            return None
        res_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
        cv2.imwrite("r_match_{}.png".format(self.match_num), res_img)
        self.match_num += 1
        return True 

if __name__ == "__main__":
    img_uniqulo_logo = cv2.imread("uniqlo.jpeg")
    img_gu_logo = cv2.imread("GU.png")
    img_seven_logo = cv2.imread("seven_and_i.png")
    img_map_1 = cv2.imread("map_1.png")
    img_map_2 = cv2.imread("map_2.png")

    map_list = [img_map_1, img_map_2]
    logo_list = [img_gu_logo, img_uniqulo_logo]
    i = 0
    for img_map in map_list:
        for img_logo in logo_list:
            imap = img_map.copy()
            akaze = AKAZE(img_logo, imap)
            rx, ry, rw, rh = akaze.full_matching()
            cv2.circle(imap, (rx+rw//2, ry+rh//2), 20, (0, 0, 0), 10)
            cv2.imwrite("result_{}.png".format(i), imap)
            i += 1
